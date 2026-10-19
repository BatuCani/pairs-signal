from datetime import datetime, timezone
from uuid import uuid4

from pydantic import Field

from pair_agent.models.schemas import Schema


class TraceStep(Schema):
    type: str
    name: str
    input: dict = Field(default_factory=dict)
    output: dict = Field(default_factory=dict)
    latency_ms: float = 0
    error: str | None = None


class Trace(Schema):
    trace_id: str = Field(default_factory=lambda: str(uuid4()))
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    request: dict
    agent_version: str
    model: str | None = None
    config: dict = Field(default_factory=dict)
    source: str
    steps: list[TraceStep] = Field(default_factory=list)
    final_answer: dict = Field(default_factory=dict)
    selected_pair: list[str] | None = None
    scores: dict = Field(default_factory=dict)
    failure_modes: list[str] = Field(default_factory=list)
    latency_ms: float = 0
    tool_call_count: int = 0
    status: str = "running"
