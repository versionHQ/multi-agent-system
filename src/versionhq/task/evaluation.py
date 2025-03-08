from typing import List, Any
from typing_extensions import Self

from pydantic import BaseModel, model_validator
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from versionhq.memory.model import MemoryMetadata


class EvaluationItem(BaseModel):
    """
    A Pydantic class to store the evaluation result with scoring and suggestion based on the given criteria.
    This class will be used as a response format for the eval task.
    """
    criteria: str
    suggestion: str
    score: float
    weight: int


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
                if new_res and new_res[0]['data']:
                    c = new_res[0]['data']['task_output']
                    w = new_res[len(new_res)-1]['data']['task_output'] if new_res[len(new_res)-1]['metadata']['score'] < new_res[0]['metadata']['score'] else ""
                    shot_prompt = SHOTS.format(c=c, w=w)

        return shot_prompt


    def _normalize_df(self) -> pd.DataFrame:
        """
        Creates a pandas DataFrame from a list of EvaluationItem objects containing 'weight' and 'score' columns, and normalizes them using MinMaxScaler.

        Args:
            items: A list of EvaluationItem objects.

        Returns:
            A pandas DataFrame with normalized 'weight' and 'score' columns, or an empty DataFrame if the input is empty.
        """
        if not self.items:
            return pd.DataFrame()

        data = { 'weight': [item.weight for item in self.items], 'score': [item.score for item in self.items] }
        df = pd.DataFrame(data)

        scaler = MinMaxScaler(feature_range=(0, 1))
        df[['weight', 'score']] = scaler.fit_transform(df[['weight', 'score']])

        return df


    @property
    def aggregate_score(self) -> int | float:
        if not self.items:
            return 0

        df = self._normalize_df()
        df['weighted_score'] = df['weight'] * df['score']
        aggregate_score = round(df['weighted_score'].sum(), 3)
        return aggregate_score


    @property
    def suggestion_summary(self) -> str | None:
        """Returns a summary of the suggestions"""

        if not self.items:
            return None

        summary = ""
        for item in self.items:
            summary += f"{item.suggestion}, "

        return summary
