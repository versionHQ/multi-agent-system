---
tags:
  - Task Graph
---

# Task Output

<class>`class` versionhq.task.model.<bold>TaskOutput<bold></class>

A Pydantic class to store and manage results of `Task`.

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

Ref. <a href="/core/task/reference/#taskoutput">List of variables and class methods</a>
