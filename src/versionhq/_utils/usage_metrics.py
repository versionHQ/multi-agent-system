import uuid
import enum
import datetime
from typing import Dict, List
from typing_extensions import Self

from pydantic import BaseModel, UUID4, InstanceOf


class ErrorType(enum.Enum):
    FORMAT = 1
    TOOL = 2
    API = 3
    OVERFITTING = 4
    HUMAN_INTERACTION = 5


class UsageMetrics(BaseModel):
    """A Pydantic model to manage token usage, errors, job latency."""

    id: UUID4 = uuid.uuid4() # stores task id or task graph id
    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    successful_requests: int = 0
    total_errors: int = 0
    error_breakdown: Dict[ErrorType, int] = dict()
    latency: float = 0.0  # in ms

    def record_token_usage(self, token_usages: List[Dict[str, int]]) -> None:
        """Records usage metrics from the raw response of the model."""

        if token_usages:
            for item in token_usages:
                self.total_tokens += int(item["total_tokens"]) if "total_tokens" in item else 0
                self.completion_tokens += int(item["completion_tokens"])  if "completion_tokens" in item else 0
                self.prompt_tokens += int(item["prompt_tokens"]) if "prompt_tokens" in item else 0


    def record_errors(self, type: ErrorType = None) -> None:
        self.total_errors += 1
        if type:
            if type in self.error_breakdown:
                self.error_breakdown[type] += 1
            else:
                self.error_breakdown[type] = 1


    def record_latency(self, start_dt: datetime.datetime, end_dt: datetime.datetime) -> None:
        self.latency += round((end_dt - start_dt).total_seconds() * 1000, 3)


    def aggregate(self, metrics: InstanceOf["UsageMetrics"]) -> Self:
        if not metrics:
            return self

        self.total_tokens += metrics.total_tokens if metrics.total_tokens else 0
        self.prompt_tokens += metrics.prompt_tokens if metrics.prompt_tokens else 0
        self.completion_tokens += metrics.completion_tokens if metrics.completion_tokens else 0
        self.successful_requests += metrics.successful_requests  if metrics.successful_requests else 0
        self.total_errors += metrics.total_errors if metrics.total_errors else 0
        self.latency += metrics.latency if metrics.latency else 0.0
        self.latency = round(self.latency, 3)

        if metrics.error_breakdown:
            for k, v in metrics.error_breakdown.items():
                if self.error_breakdown and k in self.error_breakdown:
                    self.error_breakdown[k] += int(v)
                else:
                    self.error_breakdown.update({ k: v })

        return self
