import copy
import json
from time import perf_counter

from pair_agent.agent.base import ResearchAgent
from pair_agent.agent.prompts import SYSTEM_PROMPT
from pair_agent.integrations.model_client import ModelClient
from pair_agent.models.enums import FailureMode
from pair_agent.models.schemas import AgentResult
from pair_agent.tools.registry import FinishArguments, ToolRegistry


class LLMResearchAgent(ResearchAgent):
    version = "llm-v1"

    def __init__(self, config, provider, client: ModelClient, universe=None, sink=None):
        super().__init__(config, provider, universe, sink)
        self.client, self.model = client, client.model

    def investigate(self, request, analyzer, trace_id):
        registry = ToolRegistry(analyzer, request, self.universe)
        messages = [dict(role="system", content=SYSTEM_PROMPT),
                    dict(role="user", content=json.dumps(dict(request=request.model_dump(),
                         universe=[s.model_dump() for s in self.universe], eligible_pairs=registry.allowed)))]
        selected, failures, finished = [], [], False
        for _ in range(self.config.max_model_steps):
            started = perf_counter()
            try:
                message = self.client.complete(messages, registry.schemas())
                if not isinstance(message, dict) or message.get("role") != "assistant":
                    raise ValueError("model response must be an assistant message")
                calls = message.get("tool_calls") or []
                if not isinstance(calls, list) or len(calls) > 8:
                    raise ValueError("invalid tool-call batch")
                message = {k: v for k, v in message.items() if k in {"role", "content", "tool_calls"}}
                self.sink.log_model_call(dict(messages=copy.deepcopy(messages)), dict(message=message),
                                          (perf_counter() - started) * 1000)
                messages.append(message)
                if not calls:
                    messages.append(dict(role="user", content="Use tools for evidence and finish_research to decide."))
                    continue
                for call in calls:
                    try:
                        name = call["function"]["name"]
                        arguments = json.loads(call["function"]["arguments"])
                        if name == "finish_research":
                            chosen = FinishArguments.model_validate(arguments).pairs
                            by_key = {p.key: p for p in analyzer.cache.values()}
                            if len(set(chosen)) != len(chosen) or len(chosen) > request.top_k:
                                raise ValueError("duplicate or too many selected pairs")
                            if any(key not in by_key or not by_key[key].accepted for key in chosen):
                                failures.append(FailureMode.FORCED_PAIR_SELECTION)
                                raise ValueError("select only analyzed pairs that pass deterministic acceptance gates")
                            selected = [by_key[key] for key in chosen]
                            finished = True
                            output = {"selection_frozen": chosen}
                            self.sink.log_tool_call(name, arguments, output)
                        else:
                            output = registry.execute(name, arguments)
                    except (ValueError, KeyError, TypeError) as exc:
                        output = {"error": str(exc)}
                        self.sink.log_tool_call(call.get("function", {}).get("name", "invalid_tool"), {}, output, error=str(exc))
                    messages.append(dict(role="tool", tool_call_id=call["id"], content=json.dumps(output, allow_nan=False)))
                    if finished:
                        break
                if finished:
                    break
            except (RuntimeError, ValueError, KeyError, TypeError) as exc:
                self.sink.log_model_call({}, {}, (perf_counter() - started) * 1000, error=str(exc))
                failures.append(FailureMode.MODEL_ERROR)
                break
        if not finished:
            failures.append(FailureMode.MODEL_ERROR)
        return AgentResult(trace_id=trace_id, request=request, agent_version=self.version, model=self.model,
                            candidates=list(analyzer.cache.values()), selected_pairs=selected,
                            failure_modes=list(dict.fromkeys(failures)),
                            decision="ACCEPTED AS RESEARCH CANDIDATE" if selected else "NO ROBUST PAIR FOUND")
