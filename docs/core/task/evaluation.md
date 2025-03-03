---
tags:
  - Task Graph
---

# Evaluation

<class>`class` versionhq.task.evaluate.<bold>Evaluation<bold></class>

A Pydantic class to store conditions and results of the evaluation.


### Variables

| <div style="width:120px">**Variable**</div> | **Data Type** | **Default** | **Nullable** | **Description** |
| :---            | :---  | :--- | :--- | :--- |
| **`items`**     | List[InstanceOf[EvaluationItem]]  | list() | - | Stores evaluation items. |
| **`eval_by`**   | Any   | None | True | Stores an agent evaluated the output. |


### Property

| <div style="width:120px">**Property**</div> | **Returns** | **Description** |
| :---               | :---  | :--- |
| **`aggregate_score`**   | float | Calucurates weighted average eval scores of the task output. |
| **`suggestion_summary`**  | str   | Returns summary of the suggestions. |



<hr>

## EvaluationItem

<class>`class` versionhq.task.evaluate.<bold>EvaluationItem<bold></class>

### Variables

| <div style="width:120px">**Variable**</div> | **Data Type** | **Default** | **Nullable** | **Description** |
| :---            | :---  | :--- | :--- | :--- |
| **`criteria`**     | str  | None | False | Stores evaluation criteria given by the client. |
| **`suggestion`**   | str   | None | True | Stores suggestion on improvement from the evaluator agent. |
| **`score`**   | float   | None | True | Stores the score on a 0 to 1 scale. |


<hr>

## Usage

Evaluator agents will evaluate the task output based on the given criteria, and store the results in the `TaskOutput` object.


```python
import versionhq as vhq
from pydantic import BaseModel

class CustomOutput(BaseModel):
    test1: str
    test2: list[str]

task = vhq.Task(
    description="Research a topic to teach a kid aged 6 about math.",
    pydantic_output=CustomOutput,
    should_evaluate=True, # triggers evaluation
    eval_criteria=["uniquness", "audience fit",],

)
res = task.execute()

assert isinstance(res.evaluation, vhq.Evaluation)
assert [item for item in res.evaluation.items if item.criteria == "uniquness" or item.criteria == "audience fit"]
assert res.evaluation.aggregate_score is not None
assert res.evaluation.suggestion_summary is not None
```

An `Evaluation` object provides scores for the given criteria.

For example, it might indicate a `uniqueness` score of 0.56, an `audience fit` score of 0.70, and an `aggregate score` of 0.63.
