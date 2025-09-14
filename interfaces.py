from pydantic import BaseModel
from typing import Any, List, Optional


# Strict output schema (no extra keys allowed)
class ExecutionResult(BaseModel):
    cmd: List[str]
    returncode: int
    stdout: str
    stderr: str
    json: Optional[Any] = None

    # Pydantic v2 style; for v1 use `class Config: extra = "forbid"`
    model_config = {"extra": "forbid"}


class EnvVar(BaseModel):
    name: str
    value: str
    model_config = {"extra": "forbid"}


class EarlyStopEvaluation(BaseModel):
    should_stop: bool
    reasoning: str
