import uuid
import datetime
from enum import IntEnum
from typing import Dict, List
from typing_extensions import Self

from pydantic import BaseModel, UUID4, InstanceOf


class ErrorType(IntEnum):
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
    input_tokens: int = 0
    output_tokens: int = 0
    successful_requests: int = 0
    total_errors: int = 0
    error_breakdown: Dict[ErrorType, int] = dict()
    latency: float = 0.0  # in ms


    def record_token_usage(self, *args, **kwargs) -> None:
        """Records usage metrics from the raw response of the model."""

        if args:
            for item in args:
                match item:
                    case dict():
                        if hasattr(self, k):
                            setattr(self, k, int(getattr(self, k)) + int(v))
                    case UsageMetrics():
                        self = self.aggregate(metrics=item)
                    case _:
                        try:
                            self.completion_tokens += item.completion_tokens if hasattr(item, "completion_tokens") else 0
                            self.prompt_tokens += item.prompt_tokens if hasattr(item, "prompt_tokens") else 0
                            self.total_tokens += item.total_tokens if hasattr(item, "total_tokens") else 0
                            self.input_tokens += item.input_tokens if hasattr(item, "input_tokens") else 0
                            self.output_tokens += item.output_tokens if hasattr(item, "output_tokens") else 0
                        except:
                            pass
        if kwargs:
            for k, v in kwargs.items():
                if hasattr(self, k):
                    setattr(self, k, int(getattr(self, k)) + int(v))


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

        self.total_tokens += metrics.total_tokens
        self.prompt_tokens += metrics.prompt_tokens
        self.completion_tokens += metrics.completion_tokens
        self.input_tokens += metrics.input_tokens
        self.output_tokens += metrics.output_tokens
        self.successful_requests += metrics.successful_requests
        self.total_errors += metrics.total_errors
        self.latency += metrics.latency
        self.latency = round(self.latency, 3)

        if metrics.error_breakdown:
            for k, v in metrics.error_breakdown.items():
                if self.error_breakdown and k in self.error_breakdown:
                    self.error_breakdown[k] += int(v)
                else:
                    self.error_breakdown.update({ k: v })

        return self
