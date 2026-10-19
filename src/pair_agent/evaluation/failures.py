from pair_agent.models.enums import FailureMode as F
from pair_agent.models.schemas import AgentResult


def classify_failures(result: AgentResult) -> list[F]:
    failures = list(result.failure_modes)
    for pair in result.candidates:
        failures.extend(pair.failure_modes)
    if any(not p.accepted for p in result.selected_pairs):
        failures.append(F.FORCED_PAIR_SELECTION)
    if result.tool_call_count > max(30, len(result.candidates) * 18):
        failures.append(F.UNNECESSARY_TOOL_CALLS)
    return list(dict.fromkeys(failures))
