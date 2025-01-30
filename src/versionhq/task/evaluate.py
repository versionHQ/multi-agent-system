from typing import List, Optional, Dict, Any
from typing_extensions import Self

from pydantic import BaseModel, Field, InstanceOf, model_validator

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
    score: int | float

    def _convert_score_to_score_format(self, weight: int = 1) -> ScoreFormat | None:
        if self.score and isinstance(self.score, (int, float)):
            return ScoreFormat(rate=self.score, weight=weight)

        else: return None



class Evaluation(BaseModel):
    # expected_outcome: Optional[str] = Field(default=None, description="human input on expected outcome")
    items: List[EvaluationItem] = []
    latency: int = Field(default=None, description="seconds")
    tokens: int = Field(default=None, description="tokens consumed")
    responsible_agent: Any = Field(default=None, description="store agent instance that evaluates the outcome")

    @model_validator(mode="after")
    def set_up_responsible_agent(self) -> Self:
        from versionhq.agent.default_agents import task_evaluator
        self.responsible_agent = task_evaluator
        return self


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
