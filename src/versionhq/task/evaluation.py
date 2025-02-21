from typing import List, Optional, Dict, Any
from typing_extensions import Self

from pydantic import BaseModel, model_validator

from versionhq.memory.model import MemoryMetadata

"""
Evaluate task output from accuracy, token consumption, and latency perspectives, and mark the score from 0 to 1.
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

    def __init__(self, config: Optional[Dict[str, ScoreFormat]] = None):
        self.config = config

        if self.config:
            for k, v in self.config.items():
                if isinstance(v, ScoreFormat):
                    setattr(self, k, v)


    def result(self) -> float:
        aggregate_score, denominator = 0, 0

        for k, v in self.__dict__.items():
            aggregate_score += v.aggregate
            denominator += v.weight

        if denominator == 0:
            return 0

        return round(aggregate_score / denominator, 3)


class EvaluationItem(BaseModel):
    """
    A Pydantic class to store the evaluation result with scoring and suggestion based on the given criteria.
    """
    criteria: str
    suggestion: str
    score: float

    def _format_score(self, weight: int = 1) -> ScoreFormat | None:
        if self.score and isinstance(self.score, float):
            return ScoreFormat(rate=self.score, weight=weight)

        else: return None


class Evaluation(BaseModel):
    """
    A Pydantic class to handle evaluation of the task output.
    """

    items: List[EvaluationItem] = []
    eval_by: Any = None


    @model_validator(mode="after")
    def set_up_evaluator(self) -> Self:
        from versionhq.agent.inhouse_agents import vhq_task_evaluator
        self.eval_by = vhq_task_evaluator
        return self


    def _create_memory_metadata(self) -> MemoryMetadata:
        """
        Create and store evaluation results in the memory metadata
        """
        eval_by = self.eval_by.key # saving memory
        score = self.aggregate_score
        eval_criteria = ", ".join([item.criteria for item in self.items]) if self.items else None
        suggestion = self.suggestion_summary
        memory_metadata = MemoryMetadata(eval_by=eval_by, score=score, eval_criteria=eval_criteria, suggestion=suggestion)
        return memory_metadata


    def _draft_fsl_prompt(self, task_description: str = None) -> str | None:
        """
        Search competitive and weak cases in the past and draft few shot learning prompt.
        """
        from versionhq.task.TEMPLATES.Description import SHOTS
        shot_prompt = None

        if self.eval_by.long_term_memory:
            res = self.eval_by.long_term_memory.search(query=task_description, latest_n=10)

            if res:
                new_res = filter(lambda x: "score" in x["metadata"], res)
                new_res = list(new_res)
                new_res.sort(key=lambda x: x["metadata"]["score"], reverse=True)
                if new_res[0]['data']:
                    c = new_res[0]['data']['task_output']
                    w = new_res[len(new_res)-1]['data']['task_output'] if new_res[len(new_res)-1]['metadata']['score'] < new_res[0]['metadata']['score'] else ""
                    shot_prompt = SHOTS.format(c=c, w=w)

        return shot_prompt


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
            score_format = item._format_score()
            aggregate_score += score_format.aggregate if score_format else 0
            denominator += score_format.weight if score_format else 0

        if denominator == 0:
            return 0

        return round(aggregate_score / denominator, 2)


    @property
    def suggestion_summary(self) -> str | None:
        """
        Returns a summary of the suggestions
        """
        if not self.items:
            return None

        summary = ""
        for item in self.items:
            summary += f"{item.suggestion}, "

        return summary
