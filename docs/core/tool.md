---
tags:
  - Utilities
---

# Tool

<class>`class` versionhq.tool.model.<bold>Tool<bold></class>

A Pydantic class to store the tool object.


## Quick Start

By defining the function, you can let the agent start to use it when they get an approval.

```python
import versionhq as vhq

def demo_func(message: str) -> str:
	return message + "_demo"

my_tool = vhq.Tool(func=demo_func)
res = my_tool.run(params=dict(message="Hi!"))

assert res == "Hi!_demo"
```

e.g. Build an agent with a simple tool

The tool result will be considered in the context when the agent call LLM.

```python
import versionhq as vhq

def demo_func() -> str:
	return "demo"

my_tool = vhq.Tool(func=demo_func)

agent = vhq.Agent(
	role="Tool Handler",
	goal="efficiently use the given tools",
	tools=[my_tool, ]
)
assert agent.tools == [my_tool]
```

### ToolSet

<class>`class` versionhq.tool.model.<bold>ToolSet<bold></class>

To add args to the tool and record the usage, use `toolset` instance.

```python
from versionhq import Tool, ToolSet, Agent

def demo_func(message: str) -> str:
	return message + "_demo"

tool_a = Tool(func=demo_func)
toolset = ToolSet(tool=tool_a, kwargs={"message": "Hi"})

agent = Agent(
	role="Tool Handler",
	goal="efficiently use the given tools",
	tools=[toolset,]
)
assert agent.tools == [toolset]
```

<hr />

## Customization

### Tool name

`[var]`<bold>`name: Optional[str] = None`</bold>

By default, the tool name will be set as a function name, but you can also define a specific name for your tool.

```python
from versionhq import Tool

def demo_func() -> str:
	return "demo"

my_tool = Tool(func=demo_func)

assert my_tool.name == "demo_func"
```

```python
from versionhq import Tool

my_tool = Tool(name="my empty tool", func=lambda x: x)

assert my_tool.name == "my empty tool"
```

Tool names are used to call cached tools later.


## Tool Execution

`[var]`<bold>`[REQUIRED] func: [Callable | Any] = None`</bold>

`[abstract class method]`<bold>`_run(self, *args, **kwargs) -> Any`</bold>

Simple function calling can be handled using an abstract class method `_run()`.

Following is the simplest way to define a quick function and execute it with the method.

```python
from versionhq import Tool

my_tool = Tool(func=lambda x: f"demo-{x}")
res = my_tool._run(x="TESTING")

assert res == "demo-TESTING"
```

Another way to define the logic is to add a function to the `func` field.

**Functions** are useful for defining more complex execution flows and reusing among multiple tools.

```python
from versionhq import Tool

def demo_func() -> str:
	"""...some complex execution..."""
	return "demo"

my_tool = Tool(func=demo_func)
res = my_tool._run()

assert res == "demo"
```

You can also pass parameters using the class method.

```python
from versionhq import Tool

def demo_func(message: str) -> str:
	return message + "_demo"

my_tool = Tool(func=demo_func)
res = my_tool._run(message="Hi!")

assert res == "Hi!_demo"
```

### Custom tool execution

`[abstract class method]`<bold>`_run**(self, *args, **kwargs) -> Any`</bold>

You can also use the class method to execute your custom tool inherited from `Tool` instance.

```python
from typing import Callable
from versionhq import Tool

class MyCustomTool(Tool):
  name: str = "custom tool"
  func: Callable

my_custom_tool = MyCustomTool(func=lambda x: len(x))
res = my_custom_tool._run(["demo1", "demo2"])

assert res == 2
```

### Cached tool execution

`[class_method]`<bold>`run(self, params: Dict[str, Any]) -> Any`</bold>

To use cached tools, call the class method `run` instead of _run.

```python
from typing import List, Any, Callable
from versionhq import Tool

class CustomTool(Tool):
    name: str = "custom tool"
    func: Callable

def demo_func(demo_list: List[Any]) -> int:
    return len(demo_list)

my_tool = CustomTool(func=demo_func)
res = my_tool.run(params=dict(demo_list=["demo1", "demo2"]))

assert res == 2
assert isinstance(my_tool.tool_handler, vhq.ToolHandler)
```

*Reference: <bold>`ToolHandler` class</bold>


## Cache

`[var]`<bold>`cache_function: Callable[..., Any] = None`</bold>

Define a cache function to call.

`[var]`<bold>`cache_handler: InstanceOf[CacheHandler] = None`</bold>

Define how to handle cache.

`[var]`<bold>`should_cache: bool = True`</bold>

Define if the tool name and arguments should be cached or not.

* Reference: <bold>`Cache Handler` class</bold>

## Function Calling LLM

To use the tools with LLM, make sure the LLM supports function calling, then add the tool to the agent or task.

### 1. Use the agent's tools

When the agent has tools, the tools will be applicable across multiple tasks that the agent will handle with approval of `can_use_agent_tools` on Task instance.

i.e., Return the agent’s tools result as a final result:

```python
from versionhq import Tool, Agent, Task

my_tool = Tool(name="demo tool", func=lambda x: "demo func")

agent = Agent(
	role="demo",
	goal="execute tools",
	func_calling_llm="gpt-4o",
	tools=[my_tool]
)

task = Task(
	description="execute tools",
	can_use_agent_tools=True, # if False, the agent's tools will NOT be called.
	tool_res_as_final=True
)

res = task.execute(agent=agent)
assert res == "demo func"
```

When the function calling LLM is not provided, we use the main model or default model `gpt-4o` .

```python
from versionhq import Tool, Agent, Task

def demo_func(): return "demo func"
my_tool = Tool(name="demo tool", func=demo_func)

agent = Agent(
	role="demo",
	goal="execute the given tools",
	llm="gemini-2.0", # this model will be set as a function calling LLM.
	tools=[my_tool]
)

task = Task(
	description="execute the given tools",
	can_use_agent_tools=True,
	tool_res_as_final=True
)

res = task.execute(agent=agent)
assert res.tool_output == "demo func"
```

```python
from versionhq import Tool, Agent, Task

my_tool = Tool(name="demo tool", func=lambda x: "demo func")

agent = Agent(
	role="Demo Tool Handler",
	goal="execute tools",
	tools=[my_tool]
)

task = Task(
	description="execute tools",
	can_use_agent_tools=True,
	tool_res_as_final=True
)

res = task.execute(agent=agent)
assert res.tool_output == "demo func"
assert agent.key in task.processed_agents
```

<hr />

**Function calling LLM**

By default, the agent will prioritize the given `func_calling_llm` over its main `llm` when it uses tools.

When you build the agent, it will check if the model acutally supports function callings, and if not,  `func_calling_llm` will be switched to main `llm` or default model.

If you want to see if the model of your choice supports function calling <bold>explicitly</bold>, run the following:

```python
from versionhq.llm.model import LLM
llm = LLM(model="<MODEL_NAME_OF_YOUR_CHOICE>")
res = llm._supports_function_calling()

assert type(res) == bool
```

### 2. Add tools to the task

This is a more explicit way to call tools on a specific task.

Note your agent will NOT own the tool after the task execution.

```python
from versionhq import Tool, ToolSet, Task, Agent

def random_func(message: str) -> str:
	return message + "_demo"

tool = Tool(name="tool", func=random_func)

tool_set = ToolSet(
	tool=tool,
	kwargs=dict(message="empty func")
)

agent = Agent(
	role="Tool Handler",
	goal="execute tools"
)

task = Task(
	description="execute the function",
	tools=[tool_set,], # use ToolSet to call args
	tool_res_as_final=True
)

res = task.execute(agent=agent)
assert res == "empty func_demo"
```

## Decorator

`[decorator]`<bold>`@tool[name: str]: Callable[..., Any] -> Any`</bold>

When you want to use an exsiting function as a tool, you can simply add a decorator to the function.

```python
from versionhq.tool.decorator import tool

@tool("demo")
def my_tool(test_words: str) -> str:
"""Test a tool decorator."""
return test_words

assert my_tool.name == "demo"
assert "Tool: demo" in my_tool.description and "'test_words': {'description': '', 'type': 'str'" in my_tool.description
assert my_tool.func("testing") == "testing"
```
