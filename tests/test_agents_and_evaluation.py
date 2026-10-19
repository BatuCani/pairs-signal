import json
from pathlib import Path

import pytest

from pair_agent.agent.heuristic_agent import HeuristicResearchAgent
from pair_agent.agent.llm_agent import LLMResearchAgent
from pair_agent.evaluation.dataset import export_failure_dataset, export_training_dataset, load_tasks, select_traces
from pair_agent.evaluation.evaluator import evaluate
from pair_agent.evaluation.metrics import aggregate
from pair_agent.integrations.model_client import ModelClient, OpenAICompatibleModelClient
from pair_agent.models.schemas import AgentResult, ResearchRequest
from pair_agent.research.candidate_selector import candidate_pairs, find_peer_candidates, resolve_request
from pair_agent.research.pair_analyzer import PairAnalyzer
from pair_agent.research.scoring import score_pair
from pair_agent.data.universe import default_universe
from pair_agent.tools.registry import ToolRegistry
from pair_agent.tracing.schemas import Trace


def test_peer_selection_and_natural_language():
    universe = default_universe()
    assert "PEP" in find_peer_candidates("KO", universe)
    assert "NVDA" not in find_peer_candidates("KO", universe)
    request = resolve_request(ResearchRequest(request="Find three pairs among major US banks."), universe)
    assert request.top_k == 3
    assert set(request.tickers) == {"JPM", "BAC", "C", "WFC", "GS", "MS"}
    assert candidate_pairs(ResearchRequest(tickers=["KO", "NVDA"]), universe) == []


def test_full_synthetic_vertical_slice(config, provider):
    result = HeuristicResearchAgent(config, provider).research(ResearchRequest(ticker="KO"))
    assert result.selected_pairs
    assert result.selected_pairs[0].key == "KO/PEP"
    assert result.selected_pairs[0].test is not None
    assert 0 <= result.evaluation.overall_score <= 1
    traces = list((config.data_dir / "traces").glob("*.jsonl"))
    assert len(traces) == 1
    trace = Trace.model_validate_json(traces[0].read_text())
    assert trace.trace_id == result.trace_id
    assert trace.selected_pair == ["KO", "PEP"]
    assert trace.tool_call_count == result.tool_call_count
    assert trace.scores["overall_score"] == result.evaluation.overall_score
    assert "NaN" not in traces[0].read_text()
    assert any(s.name == "heldout_backtest" for s in trace.steps)


def test_agent_can_reject_all_candidates(config, provider):
    result = HeuristicResearchAgent(config, provider).research(ResearchRequest(tickers=["NVDA", "AMD"]))
    assert not result.selected_pairs
    assert result.decision == "NO ROBUST PAIR FOUND"
    assert any("HIGH_CORRELATION_NO_COINTEGRATION" in p.failure_modes for p in result.candidates)
    assert result.evaluation.overall_score < 1


def test_single_symbol_universe_no_forced_selection(config, provider):
    result = HeuristicResearchAgent(config, provider).research(ResearchRequest(tickers=["KO"]))
    assert not result.selected_pairs
    assert result.evaluation.overall_score == 0


def test_scoring_ignores_heldout_and_rejects_cost_sensitivity(config, provider):
    pair = PairAnalyzer(config, provider).analyze_pair("KO", "PEP")
    assert pair.accepted
    before = pair.score
    pair.test = pair.validation.model_copy(update={"total_return": -0.9, "sharpe_ratio": -20})
    assert score_pair(pair, config).score == before
    pair.validation_stressed = pair.validation_stressed.model_copy(update={"total_return": -.1})
    pair = score_pair(pair, config)
    assert not pair.accepted
    assert "TRANSACTION_COST_SENSITIVE" in pair.failure_modes


def test_evaluator_penalizes_forced_selection(config, provider):
    pair = PairAnalyzer(config, provider).analyze_pair("NVDA", "AMD")
    result = AgentResult(trace_id="bad", request=ResearchRequest(), agent_version="bad", selected_pairs=[pair], candidates=[pair])
    evaluation = evaluate(result, config)
    assert evaluation.overall_score == 0
    assert "FORCED_PAIR_SELECTION" in evaluation.failure_modes


def test_trace_export_filters_success_and_failure(config, provider, tmp_path):
    agent = HeuristicResearchAgent(config, provider)
    success = agent.research(ResearchRequest(ticker="KO"))
    failure = agent.research(ResearchRequest(tickers=["NVDA", "AMD"]))
    directory = config.data_dir / "traces"
    assert len(select_traces(directory)) == 2
    output = tmp_path / "training.jsonl"
    assert export_training_dataset(directory, output, min_score=success.evaluation.overall_score) == 1
    row = json.loads(output.read_text())
    assert row["messages"][0]["role"] == "user"
    assert row["metadata"]["source"] == "FrameMarketData"
    assert export_failure_dataset(directory, tmp_path / "failed.jsonl") == 1
    assert json.loads((tmp_path / "failed.jsonl").read_text())["metadata"]["trace_id"] == failure.trace_id


def test_starter_dataset_has_twenty_valid_tasks():
    assert len(load_tasks(Path("data/tasks.jsonl"))) == 20


def test_aggregate_handles_empty():
    assert aggregate([]) == {"tasks": 0}


class ScriptedClient(ModelClient):
    model = "offline-scripted-test"

    def __init__(self, actions):
        self.actions = iter(actions)

    def complete(self, messages, tools):
        name, arguments = next(self.actions)
        return dict(role="assistant", content=None,
                    tool_calls=[dict(id=f"call_{len(messages)}", type="function",
                                     function=dict(name=name, arguments=json.dumps(arguments)))])


def test_llm_uses_real_tools_and_finishes(config, provider):
    client = ScriptedClient([("calculate_correlation", dict(ticker_a="KO", ticker_b="PEP")),
                             ("analyze_pair", dict(ticker_a="KO", ticker_b="PEP")),
                             ("finish_research", {"pairs": ["KO/PEP"]})])
    result = LLMResearchAgent(config, provider, client).research(ResearchRequest(ticker="KO"))
    assert result.selected_pairs[0].key == "KO/PEP"
    assert result.model_call_count == 3
    assert result.selected_pairs[0].test is not None
    trace = select_traces(config.data_dir / "traces")[0]
    assert len([s for s in trace.steps if s.type == "model_call"]) == 3
    output = config.data_dir / "llm-export.jsonl"
    assert export_training_dataset(config.data_dir / "traces", output, min_score=0) == 1
    messages = json.loads(output.read_text())["messages"]
    pending = set()
    for message in messages:
        if message["role"] == "assistant":
            assert not pending
            pending.update(c["id"] for c in message.get("tool_calls") or [])
        if message["role"] == "tool":
            pending.remove(message["tool_call_id"])
    assert not pending


def test_llm_cannot_invent_selected_pair(config, provider):
    client = ScriptedClient([("finish_research", {"pairs": ["KO/PEP"]}),
                             ("finish_research", {"pairs": []})])
    result = LLMResearchAgent(config, provider, client).research(ResearchRequest(ticker="KO"))
    assert not result.selected_pairs
    assert "FORCED_PAIR_SELECTION" in result.failure_modes


def test_model_error_is_traced_and_does_not_force_pair(config, provider):
    class BrokenClient(ModelClient):
        model = "offline-broken"
        def complete(self, messages, tools):
            raise RuntimeError("simulated endpoint failure")
    result = LLMResearchAgent(config, provider, BrokenClient()).research(ResearchRequest(ticker="KO"))
    assert not result.selected_pairs
    assert "MODEL_ERROR" in result.failure_modes


def test_tool_registry_blocks_unknown_pairs_and_arguments(config, provider):
    registry = ToolRegistry(PairAnalyzer(config, provider), ResearchRequest(ticker="KO"), default_universe())
    with pytest.raises(ValueError, match="outside"):
        registry.execute("analyze_pair", dict(ticker_a="KO", ticker_b="NVDA"))
    with pytest.raises(ValueError):
        registry.execute("calculate_correlation", dict(ticker_a="KO", ticker_b="PEP", train_end="2099"))
    with pytest.raises(ValueError):
        registry.execute("execute_trade", {})


def test_model_endpoint_rejects_remote_plaintext_and_missing_credentials():
    with pytest.raises(ValueError):
        OpenAICompatibleModelClient("http://example.com/v1", "key", "model")
    with pytest.raises(ValueError):
        OpenAICompatibleModelClient("https://example.com/v1", "", "model")
