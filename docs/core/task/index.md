---
tags:
  - Task Graph
---

# Task

<class>`class` versionhq.task.model.<bold>Task<bold></class>

A class to store and manage information for individual tasks, including their assignment to agents or agent networks, and dependencies via a node-based system that tracks conditions and status.

Ref. Node / Edge / <a href="/core/task-graph">TaskGraph</a> class

<hr />

## Quick Start

Create a task by defining its description in one simple sentence. The `description` will be used for task prompting later.

Each task will be assigned a unique ID as an identifier.

```python
import versionhq as vhq

task = vhq.Task(description="MY AMAZING TASK")

import uuid
assert uuid.UUID(str(task.id), version=4)
```


And you can simply execute the task by calling `.execute()` function.

```python
import versionhq as vhq

task = vhq.Task(description="MY AMAZING TASK")
res = task.execute()

assert isinstance(res, vhq.TaskOutput) # Generates TaskOutput object
assert res.raw and res.json # By default, TaskOutput object stores output in plane text and json formats.
assert task.processed_agents is not None # Agents will be automatically assigned to the given task.
```

<hr />

## Structured Response

By default, agents will generate plane text and JSON outputs, and store them in the `TaskOutput` object.

* Ref. <a href="/core/task/task-output">`TaskOutput`</a> class

But you can choose to generate Pydantic class or specifig JSON object as response.

<hr />

**1. Pydantic**

`[var]`<bold>`pydantic_output: Optional[Type[BaseModel]] = None`</bold>

Create and add a `custom Pydantic class` as a structured response format to the `pydantic_output` field.

The custom class can accept **one layer of a nested child** as you can see in the following code snippet:

```python
import versionhq as vhq
from pydantic import BaseModel
from typing import Any


# 1. Define Pydantic class with a description (optional), annotations and field names.
class Demo(BaseModel):
    """
    A demo pydantic class to validate the outcome with various nested data types.
    """
    demo_1: int
    demo_2: float
    demo_3: str
    demo_4: bool
    demo_5: list[str]
    demo_6: dict[str, Any]
    demo_nest_1: list[dict[str, Any]] # 1 layer of nested child is ok.
    demo_nest_2: list[list[str]]
    demo_nest_3: dict[str, list[str]]
    demo_nest_4: dict[str, dict[str, Any]]
    # error_1: list[list[dict[str, list[str]]]] # <- Trigger 400 error due to 2+ layers of nested child.
    # error_2: InstanceOf[AnotherPydanticClass] # <- Trigger 400 error due to non-typing annotation.
    # error_3: list[InstanceOf[AnotherPydanticClass]] # <- Trigger 400 error due to non-typing annotation as a nested child.

# 2. Define a task
task = vhq.Task(
    description="generate random output that strictly follows the given format",
    pydantic_output=Demo,
)

# 3. Execute
res = task.execute()

assert isinstance(res, vhq.TaskOutput)
assert res.raw and res.json
assert isinstance(res.raw, str) and isinstance(res.json_dict, dict)
assert [
    getattr(res.pydantic, k) and v.annotation == Demo.model_fields[k].annotation
    for k, v in res.pydantic.model_fields.items()
]
```

**2. JSON**

`[var]`<bold>`response_fields: List[InstanceOf[ResponseField]] = None`</bold>

Similar to Pydantic, JSON output structure can be defined by using a list of `ResponseField` objects.

The following code snippet demonstrates how to use `ResponseField` to generate output with a maximum of one level of nesting.

Custom JSON outputs can accept **one layer of a nested child**.

**[NOTES]**

- `demo_response_fields` in the following case is identical to the previous Demo class, except that titles are specified for nested fields.

- Agents generate JSON output by default, whether or not `response_fields` are used.

- However, response_fields are REQUIRED to specify JSON key titles and data types.

```python
import versionhq as vhq

# 1. Define a list of ResponseField objects.
demo_response_fields = [
    # no nesting
    vhq.ResponseField(title="demo_1", data_type=int),
    vhq.ResponseField(title="demo_2", data_type=float),
    vhq.ResponseField(title="demo_3", data_type=str),
    vhq.ResponseField(title="demo_4", data_type=bool),
    vhq.ResponseField(title="demo_5", data_type=list, items=str),
    vhq.ResponseField(
        title="demo_6",
        data_type=dict,
        properties=[vhq.ResponseField(title="demo-item", data_type=str)]
    ),
    # nesting
    vhq.ResponseField(
        title="demo_nest_1",
        data_type=list,
        items=dict,
        properties=([
            vhq.ResponseField(
                title="nest1",
                data_type=dict,
                properties=[vhq.ResponseField(title="nest11", data_type=str)]
            )
        ])
    ),
    vhq.ResponseField(title="demo_nest_2", data_type=list, items=list),
    vhq.ResponseField(title="demo_nest_3", data_type=dict, properties=[
        vhq.ResponseField(title="nest1", data_type=list, items=str)
    ]),
    vhq.ResponseField(title="demo_nest_4", data_type=dict, properties=[
        vhq.ResponseField(
            title="nest1",
            data_type=dict,
            properties=[vhq.ResponseField(title="nest12", data_type=str)]
        )
    ])
]


# 2. Define a task
task = vhq.Task(
    description="Output random values strictly following the data type defined in the given response format.",
    response_fields=demo_response_fields
)


# 3. Execute
res = task.execute()

assert isinstance(res, vhq.TaskOutput) and res.task_id is task.id
assert res.raw and res.json and res.pydantic is None
assert [v and type(v) == task.response_fields[i].data_type for i, (k, v) in enumerate(res.json_dict.items())]
```

* Ref. <a href="/core/task/response-field">`ResponseField`</a> class

<hr />

**Structuring reponse format**

- Higlhy recommends assigning agents optimized for `gemini-x` or `gpt-x` to produce structured outputs with nested items.

- To generate response with more than 2 layers of nested items, seperate them into multipe tasks or utilize nodes.

The following case demonstrates to returning a `Main` class that contains a nested `Sub` class.

**[NOTES]**

- Using `callback` functions to format the final response. (You can try other functions suitable for your use case.)

- Passing parameter: `sub` to the callback function via the `callback_kwargs` variable.

- By default, the outputs of `main_task` are automatically passed to the callback function; you do NOT need to explicitly define them.

- Callback results will be stored in the `callback_output` field of the `TaskOutput` class.


```python
import versionhq as vhq
from pydantic import BaseModel
from typing import Any

# 1. Define and execute a sub task with Pydantic output.
class Sub(BaseModel):
    sub1: list[dict[str, Any]]
    sub2: dict[str, Any]

sub_task = vhq.Task(
    description="generate random values that strictly follows the given format.",
    pydantic_output=Sub
)
sub_res = sub_task.execute()

# 2. Define a main task, callback function to format the final response.
class Main(BaseModel):
    main1: list[Any] # <= assume expecting to store Sub object in this field.
    # error_main1: list[InstanceOf[Sub]]  # as this will trigger 400 error!
    main2: dict[str, Any]

def format_response(sub: InstanceOf[Sub], main1: list[Any], main2: dict[str, Any]) -> Main:
    main1.append(sub)
    main = Main(main1=main1, main2=main2)
    return main

# 3. Execute
main_task = vhq.Task(
    description="generate random values that strictly follows the given format",
    pydantic_output=Main,
    callback=format_response,
    callback_kwargs=dict(sub=Sub(sub1=sub_res.pydantic.sub1, sub2=sub_res.pydantic.sub2)),
)
res = main_task.execute(context=sub_res.raw) # [Optional] Adding sub_task as a context.

assert [item for item in res.callback_output.main1 if isinstance(item, Sub)]
```

To automate these manual setups, refer to <a href="/core/agent-network">AgentNetwork</a> class.

<hr />

## Prompting

`[class method]`<bold>`prompt(self, model_provider: str = None, context: Optional[Any] = None) -> str`</bold>

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
task_prompt = task._prompt(context=context)

assert sub_res.to_context_prompt() in task_prompt
assert sub_task_2.output and sub_task_2.output.to_context_prompt() in task_prompt  # sub tasks' outputs are included in the task prompt.
assert "context to add in string" in task_promp
assert res
```

Context can consist of `Task` objects, `TaskOutput` objects, plain text `strings`, or `lists` containing any of these.

In this scenario, `sub_task_2` executes before the main task. Its string output is then incorporated into the main task's context prompt on top of other context before the main task is executed.

<hr>

## Executing

**Agent delegation**

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
assert "vhq-Delegated-Agent" in task.processed_agents # delegated agent
assert task.delegations ==1
```

<hr>

**SYNC - ASYNC**

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

**Using tools**

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

Ref. <a href="/core/tool">Tool</a> class / <a href="/core/task/task-output">TaskOutput</a> class

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

## Callback

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

## Evaluating

`[var]`<bold>`should_evaluate: bool = False`</bold>

`[var]`<bold>`eval_criteria: Optional[List[str]] = list()`</bold>

You can turn on customized evaluations using the given criteria.

Refer <a href="/core/task/task-output">TaskOutput</a> class for details.

<hr>


## Reference

### Variables

| <div style="width:160px">**Variable**</div> | **Data Type** | **Default** | **Nullable** | **Description** |
| :---               | :---  | :--- | :--- | :--- |
| **`id`**   | UUID  | uuid.uuid4() | False | Stores task `id` as an identifier. |
| **`name`**       | Optional[str]   | None | True | Stores a task name (Inherited as `node` identifier if the task is dependent) |
| **`description`**       | str   | None | False | Required field to store a concise task description |
| **`pydantic_output`** | Optional[Type[BaseModel]] | None | True | Stores pydantic custom output class for structured response |
| **`response_fields`** | Optional[List[ResponseField]]  | list() | True | Stores JSON formats for stuructured response |
| **`tools`** |  Optional[List[ToolSet | Tool | Any]] | None | True | Stores tools to be called when the agent executes the task. |
| **`can_use_agent_tools`** |  bool | True | - | Whether to use the agent tools |
| **`tool_res_as_final`** |  bool | False | - | Whether to make the tool response a final response from the agent |
| **`execution_type`** | TaskExecutionType  | TaskExecutionType.SYNC | - | Sync or async execution |
| **`allow_delegation`** | bool  | False | - | Whether to allow the agent to delegate the task to another agent |
| **`callback`** | Optional[Callable] | None | True | Callback function to be executed after LLM calling |
| **`callback_kwargs`** | Optional[Dict[str, Any]] | dict() | True | Args for the callback function (if any)|
| **`should_evaluate`** | bool | False | - | Whether to evaluate the task output using eval criteria |
| **`eval_criteria`** | Optional[List[str]] | list() | True | Evaluation criteria given by the human client |
| **`processed_agents`** | Set[str] | set() | True | [Ops] Stores roles of the agents executed the task |
| **`tool_errors`** | int | 0 | True | [Ops] Stores number of tool errors |
| **`delegation`** | int | 0 | True | [Ops] Stores number of agent delegations |
| **`output`** | Optional[TaskOutput] | None | True | [Ops] Stores `TaskOutput` object after the execution |


### Class Methods

| <div style="width:120px">**Method**</div> |  <div style="width:300px">**Params**</div> | **Returns** | **Description** |
| :---               | :---  | :--- | :--- |
| **`execute`**  | <p>type: TaskExecutionType = None<br>agent: Optional["vhq.Agent"] = None<br>context: Optional[Any] = None</p> | InstanceOf[`TaskOutput`] or None (error) |  A main method to handle task execution. Auto-build an agent when the agent is not given. |


### Properties

| <div style="width:120px">**Property**</div> | **Returns** | **Description** |
| :---               | :---  | :--- |
| **`key`**   | str | Returns task key based on its description and output format. |
| **`summary`**  | str   | Returns a summary of the task based on its id, description and tools. |
