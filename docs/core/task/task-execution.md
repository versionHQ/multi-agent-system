# Executing Task


## Prompt Engineering

Prompts are generated automatically based on the task `description`, response format, context, agent `role`, and `goal`.


**Context**

The following snippet demonstrates how to add `context` to the prompt.

```python
import versionhq as vhq

sub_task_1 = vhq.Task(description="Run a sub demo part 1")
sub_res = sub_task_1.execute()

sub_task_2 = vhq.Task(description="Run a sub demo part 2")

task = vhq.Task(description="Run a main demo")

context = [sub_res, sub_task_2, "context to add in string"]
res = task.execute(context=context)

# Explicitly mentioned. `task.execute()` will trigger the following:
task_prompt = task._user_prompt(context=context)

assert sub_res._to_context_prompt() in task_prompt
assert sub_task_2.output and sub_task_2.output._to_context_prompt() in task_prompt  # sub tasks' outputs are included in the task prompt.
assert "context to add in string" in task_promp
assert res
```

Context can consist of `Task` objects, `TaskOutput` objects, plain text `strings`, or `lists` containing any of these.

In this scenario, `sub_task_2` executes before the main task. Its string output is then incorporated into the main task's context prompt on top of other context before the main task is executed.

<hr>

## Delegation

`[var]`<bold>`allow_delegation: bool = False`</bold>

You can assign another agent to complete the task:

```python
import versionhq as vhq

task = vhq.Task(
    description="return the output following the given prompt.",
    allow_delegation=True
)
task.execute()

assert task.output is not None
assert task.processed_agents is not None # auto assigned
assert task._delegations ==1
```

<hr>

## Sync - Async Execution

`[var]`<bold>`type: bool = False`</bold>

You can specify whether the task will be executed asynchronously.

```python
import versionhq as vhq

task = vhq.Task(
    description="Return a word: 'test'",
    type=vhq.TaskExecutionType.ASYNC # default: vhq.TaskExecutionType.SYNC
)

from unittest.mock import patch
with patch.object(vhq.Agent, "execute_task", return_value="test") as execute:
    res = task.execute()
    assert res.raw == "test"
    execute.assert_called_once_with(task=task, context=None, task_tools=list())
```

<hr>

## Tools

`[var]`<bold>`tools: Optional[List[ToolSet | Tool | Any]] = None`</bold>

`[var]`<bold>`tool_res_as_final: bool = False`</bold>


Tasks can directly store tools explicitly called by the agent.

If the results from the tool should be the final results, set `tool_res_as_final` True.

This will allow the agent to store the tool results in the `tool_output` field of `TaskOutput` object.


```python
import versionhq as vhq
from typing import Callable

def random_func(message: str) -> str:
    return message + "_demo"

tool = vhq.Tool(name="tool", func=random_func)
tool_set = vhq.ToolSet(tool=tool, kwargs=dict(message="empty func"))
task = vhq.Task(
    description="execute the given tools",
    tools=[tool_set,], # stores tools
    tool_res_as_final=True, # stores tool results in TaskOutput object
)

res = task.execute()
assert res.tool_output == "empty func_demo"
```

Ref 1. <a href="/core/tool">Tool</a> class / <a href="/core/rag-tool">RAGTool</a> class

Ref 2. <a href="/core/task/task-output">TaskOutput</a> class

<hr>

**Using agents' tools**

`[var]`<bold>`can_use_agent_tools: bool = True`</bold>

Tasks can explicitly stop/start using agent tools on top of the tools stored in the task object.

```python
import versionhq as vhq

simple_tool = vhq.Tool(name="simple tool", func=lambda x: "simple func")
agent = vhq.Agent(role="demo", goal="execute tools", tools=[simple_tool,])
task = vhq.Task(
    description="execute tools",
    can_use_agent_tools=True, # Flagged
    tool_res_as_final=True
)
res = task.execute(agent=agent)
assert res.tool_output == "simple func"
```

<hr>

## Image, Audio, File Content

Refer the content by adding an absolute file path to the content file or URL to the task object.


```python
import versionhq as vhq
from pathlib import Path

current_dir = Path(__file__).parent.parent
file_path = current_dir / "_sample/screenshot.png"
audio_path = current_dir / "_sample/sample.mp3"

task = vhq.Task(description="Summarize the given content", image=str(file_path), audio=str(audio_path))
res = task.execute(agent=vhq.Agent(llm="gemini-2.0", role="Content Interpretator"))

assert res.raw is not None
```

* Audio files are only applicable to `gemini` models.

<hr>

## Callbacks

`[var]`<bold>`callback: Optional[Callable] = None`</bold>

`[var]`<bold>`callback_kwargs: Optional[Dict[str, Any]] = dict()`</bold>

After executing the task, you can run a `callback` function with `callback_kwargs` and task output as parameters.

Callback results will be stored in `callback_output` filed of the `TaskOutput` object.

```python
import versionhq as vhq

def callback_func(condition: str, test1: str):
    return f"Result: {test1}, condition added: {condition}"

task = vhq.Task(
    description="return the output following the given prompt.",
    callback=callback_func,
    callback_kwargs=dict(condition="demo for pytest")
)
res = task.execute()

assert res and isinstance(res, vhq.TaskOutput)
assert res.task_id is task.id
assert "demo for pytest" in res.callback_output
```

<hr>
