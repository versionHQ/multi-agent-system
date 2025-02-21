
"""Test use cases on core/tool.md"""

def test_docs_core_tool_a():
    from versionhq import Tool

    def demo_func(message: str) -> str:
        return message + "_demo"

    my_tool = Tool(func=demo_func)
    res = my_tool.run(params=dict(message="Hi!"))

    assert res == "Hi!_demo"


def test_docs_core_tool_b():
    from versionhq import Tool, Agent

    def demo_func() -> str:
        return "demo"

    my_tool = Tool(func=demo_func)

    agent = Agent(
        role="Tool Handler",
        goal="efficiently use the given tools",
        tools=[my_tool, ]
    )

    assert agent.tools == [my_tool, ]
    assert agent.func_calling_llm.model is not None


def test_docs_core_tool_c():
    from versionhq import Tool, ToolSet, Agent

    def demo_func(message: str) -> str:
        return message + "_demo"

    tool_a = Tool(func=demo_func)
    toolset = ToolSet(tool=tool_a, kwargs={"message": "Hi"})
    agent = Agent(
        role="Tool Handler",
        goal="efficiently use the given tools",
        tools=[toolset]
    )

    assert agent.tools == [toolset]
    assert agent.func_calling_llm.model is not None


def test_docs_core_tool_d():
    from versionhq import Tool

    my_tool = Tool(name="my empty tool", func = lambda x: x)
    assert my_tool.name == "my empty tool" and my_tool.func is not None


    def demo_func() -> str:
        return "demo"

    my_tool_2 = Tool(func=demo_func)
    assert my_tool_2.name == "demo_func" and my_tool_2.func == demo_func


def test_docs_core_tool_e1():
    from versionhq import Tool

    my_tool = Tool(func=lambda x: f"demo-{x}" )
    res = my_tool._run(x="TESTING")
    assert res == "demo-TESTING"


def test_docs_core_tool_e2():
    from versionhq import Tool

    def demo_func() -> str:
        """...some complex execution..."""
        return "demo"

    my_tool = Tool(func=demo_func)
    res = my_tool._run()
    assert res == "demo"


def test_docs_core_tool_e3():
    from versionhq import Tool

    def demo_func(message: str) -> str:
        return message + "_demo"

    my_tool = Tool(func=demo_func)
    res = my_tool._run(message="Hi!")

    assert res == "Hi!_demo"


def test_docs_core_tool_f():
    from typing import Callable
    from versionhq import Tool

    class MyCustomTool(Tool):
        name: str = "custom tool"
        func: Callable

    my_custom_tool = MyCustomTool(func=lambda x: len(x))
    res = my_custom_tool._run(["demo1", "demo2"])
    assert res == 2


def test_docs_core_tool_g():
    from typing import List, Any, Callable
    from versionhq import Tool

    class CustomTool(Tool):
        name: str = "custom tool"
        func: Callable

    def demo_func(demo_list: List[Any]) -> int:
        return len(demo_list)

    my_tool = CustomTool(func=demo_func)
    res = my_tool.run(params=dict(demo_list=["demo1", "demo2"]))

    from versionhq import ToolHandler
    assert res == 2
    assert isinstance(my_tool.tool_handler, ToolHandler)


def test_docs_core_tool_h1():
    from versionhq import Tool, Agent, Task

    my_tool = Tool(name="demo tool", func=lambda x: "demo func")
    assert my_tool

    agent = Agent(
        role="demo",
        goal="execute tools",
        func_calling_llm="gpt-4o",
        tools=[my_tool],
        maxit=1,
        max_tokens=3000
    )
    assert agent.tools == [my_tool]

    task = Task(
        description="execute tools",
        can_use_agent_tools=True,
        tool_res_as_final=True
    )

    res = task.execute(agent=agent)
    assert res.raw == "demo func"


def test_docs_core_tool_h2():
    from versionhq import Tool, Agent, Task

    def demo_func(): return "demo func"
    my_tool = Tool(name="demo tool", func=demo_func)
    agent = Agent(
        role="demo",
        goal="execute the given tools",
        llm="gemini-2.0",
        tools=[my_tool],
        maxit=1,
        max_tokens=3000
    )

    task = Task(
        description="execute the given tools",
        can_use_agent_tools=True,
        tool_res_as_final=True
    )

    res = task.execute(agent=agent)
    assert res.raw == "demo func" and res.tool_output == "demo func"


def test_docs_core_tool_h3():
    from versionhq import Tool, Agent, Task

    my_tool = Tool(name="demo tool", func=lambda x: "demo func")

    agent = Agent(
        role="Demo Tool Handler",
        goal="execute tools",
        tools=[my_tool],
        maxit=1,
    )

    task = Task(
        description="execute tools",
        can_use_agent_tools=True,
        tool_res_as_final=True
    )

    res = task.execute(agent=agent)

    assert res.tool_output == "demo func"
    assert agent.key in task.processed_agents


def test_docs_core_tool_h4():
    from versionhq.llm.model import LLM

    llm = LLM(model="gpt_4o")
    assert isinstance(llm._supports_function_calling(), bool)


def test_docs_core_tool_h5():
    from versionhq import Tool, ToolSet, Task, Agent

    def random_func(message: str) -> str:
        return message + "_demo"

    tool = Tool(name="tool", func=random_func)

    tool_set = ToolSet(
        tool=tool,
        kwargs=dict(message="empty func")
    )
    task = Task(
        description="execute the function",
        tools=[tool_set,], # use ToolSet to call args
        tool_res_as_final=True
    )

    res = task.execute()
    assert res.tool_output == "empty func_demo"


def test_docs_core_tool_i():
    from versionhq.tool.decorator import tool

    @tool("demo")
    def my_tool(test_words: str) -> str:
        """Test a tool decorator."""
        return test_words

    assert my_tool.name == "demo"
    assert "Tool: demo" in my_tool.description and "'test_words': {'description': '', 'type': 'str'" in my_tool.description
    assert my_tool.func("testing") == "testing"
