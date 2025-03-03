---
tags:
  - Task Graph
---

# Task Output

<class>`class` versionhq.task.model.<bold>TaskOutput<bold></class>

A Pydantic class to store and manage results of `Task`.

<hr />

## Variables

| <div style="width:120px">**Variable**</div> | **Data Type** | **Default** | **Nullable** | **Description** |
| :---               | :---  | :--- | :--- | :--- |
| **`task_id`**   | UUID  | uuid.uuid4() | False | Stores task `id` as an identifier. |
| **`raw`**       | str   | None | False | Stores response in plane text format. `None` or `""` when the model returned errors.|
| **`json_dict`** | Dict[str, Any] | None | False | Stores response in JSON serializable dictionary. When the system failed formatting or executing tasks without response_fields, `{ output: <res.raw> }` will be returned. |
| **`pydantic`** | Type[`BaseModel`]  | None | True | Populates and stores Pydantic class object defined in the `pydantic_output` field. `None` if `pydantic_output` is NOT given. |
| **`tool_output`** |  Optional[Any] | None | True | Stores results from the tools of the task or agents ONLY when `tool_res_as_final` set as `True`. |
| **`callback_output`** |  Optional[Any] | None | True | Stores results from callback functions if any. |
| **`latency`** |  Optional[float] | None | True | Stores job latency in milseconds. |
| **`evaluation`** |  Optional[InstanceOf[`Evaluation`]] | None | True | Stores overall evaluations and usage of the task output. |


The following snippet demonstrates the  `TaskOutput` object when the task is all-in with Pydantic response format, callbacks, tools, and evaluation.

```python
import versionhq as vhq
from pydantic import BaseModel

class CustomOutput(BaseModel):
    test1: str
    test2: list[str]

def dummy_tool():
    return "dummy"

def summarize_response(message: str, test1: str, test2: list[str]) -> str:
    return f"""{message}: {test1}, {", ".join(test2)}"""

task = vhq.Task(
    description="Research a topic to teach a kid aged 6 about math.",
    pydantic_output=CustomOutput,
    tools=[dummy_tool],
    callback=summarize_response,
    callback_kwargs=dict(message="Hi! Here is the result: "),
    should_evaluate=True, # triggers evaluation
    eval_criteria=["Uniquness", "Fit to audience",],

)
res = task.execute()

assert res.task_id == task.id
assert res.raw
assert res.json_dict
assert res.pydantic.test1 and res.pydantic.test2
assert "Hi! Here is the result: " in res.callback_output
assert res.pydantic.test1 in res.callback_output and ", ".join(res.pydantic.test2) in res.callback_output
assert res.tool_output is None
assert res.evaluation and isinstance(res.evaluation, vhq.Evaluation)
```


## Class Methods

| <div style="width:120px">**Method**</div> | **Params** | **Returns** | **Description** |
| :---               | :---  | :--- | :--- |
| **`evaluate`**   | task: InstanceOf[`Task`]  | InstanceOf[`Evaluation`]  | Evaluates task output based on the criteria |

Ref. <a href="/core/task/evaluation">Evaluation</a> class

## Property

| <div style="width:120px">**Property**</div> | **Returns** | **Description** |
| :---               | :---  | :--- |
| **`aggregate_score`**   | float | Calucurates weighted average eval scores of the task output. |
| **`json_string`**       | str   | Returns `json_dict` in string format. |
