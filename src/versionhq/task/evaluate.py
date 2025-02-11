from typing import List, Optional, Dict, Any
from typing_extensions import Self

from pydantic import BaseModel, Field, model_validator

from versionhq.memory.model import MemoryMetadata

"""
Evaluate task output from accuracy, token consumption, latency perspectives, and mark the score from 0 to 1.
"""


class ScoreFormat:
    def __init__(self, rate: float | int = 0, weight: int = 1):
        self.rate = rate
        self.weight = weight
        self.aggregate = rate * weight


class Score:
    """
    Evaluate the score on 0 (no performance) to 1 scale.
    `rate`: Any float from 0.0 to 1.0 given by an agent.
    `weight`: Importance of each factor to the aggregated score.
    """

    def __init__(
        self,
        brand_tone: ScoreFormat = ScoreFormat(0, 0),
        audience: ScoreFormat = ScoreFormat(0, 0),
        track_record: ScoreFormat = ScoreFormat(0, 0),
        config: Optional[Dict[str, ScoreFormat]] = None
    ):
        self.brand_tone = brand_tone
        self.audience = audience
        self.track_record = track_record
        self.config = config

        if self.config:
            for k, v in self.config.items():
                if isinstance(v, ScoreFormat):
                    setattr(self, k, v)


    def result(self) -> int:
        aggregate_score, denominator = 0, 0

        for k, v in self.__dict__.items():
            aggregate_score += v.aggregate
            denominator += v.weight

        if denominator == 0:
            return 0

        return round(aggregate_score / denominator, 2)


class EvaluationItem(BaseModel):
    """
    A class to store evaluation and suggestion by the given criteria such as accuracy.
    """
    criteria: str
    suggestion: str
    score: float

    def _convert_score_to_score_format(self, weight: int = 1) -> ScoreFormat | None:
        if self.score and isinstance(self.score, float):
            return ScoreFormat(rate=self.score, weight=weight)

        else: return None


class Evaluation(BaseModel):
    items: List[EvaluationItem] = []
    latency: float = Field(default=None, description="job execution latency in seconds")
    tokens: int = Field(default=None, description="tokens consumed")
    eval_by: Any = Field(default=None, description="stores agent object that evaluates the outcome")

    @model_validator(mode="after")
    def set_up_evaluator(self) -> Self:
        from versionhq.agent.inhouse_agents import vhq_task_evaluator
        self.eval_by = vhq_task_evaluator
        return self


    def _create_memory_metadata(self) -> MemoryMetadata:
        """
        Create and store evaluation results in the memory metadata
        """
        eval_by = self.eval_by.role if self.eval_by else None
        score = self.aggregate_score
        eval_criteria = ", ".join([item.criteria for item in self.items]) if self.items else None
        suggestion = self.suggestion_summary
        memory_metadata = MemoryMetadata(eval_by=eval_by, score=score, eval_criteria=eval_criteria, suggestion=suggestion)
        return memory_metadata


    @property
    def aggregate_score(self) -> float:
        """
        Calcurate aggregate score from evaluation items.
        """
        if not self.items:
            return 0

        aggregate_score = 0
        denominator = 0

        for item in self.items:
            score_format = item._convert_score_to_score_format()
            aggregate_score += score_format.aggregate if score_format else 0
            denominator += score_format.weight if score_format else 0

        if denominator == 0:
            return 0

        return round(aggregate_score / denominator, 2)


    @property
    def suggestion_summary(self) -> str | None:
        """
        Return a summary of the suggestions
        """
        if not self.items:
            return None

        summary = ""
        for item in self.items:
            summary += f"{item.suggestion}, "

        return summary
