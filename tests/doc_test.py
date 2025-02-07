"""
Pytest use cases in the documentation.
"""

def test_quick_start():
    """index.md, quickstart.md"""

    def a():
        from versionhq import form_agent_network

        network = form_agent_network(
            task="YOUR AMAZING TASK OVERVIEW",
            expected_outcome="YOUR OUTCOME EXPECTATION",
        )
        res = network.launch()
        assert res.raw is not None


    def b():
        from pydantic import BaseModel
        from versionhq import Agent, Task

        class CustomOutput(BaseModel):
            test1: str
            test2: list[str]

        def dummy_func(message: str, test1: str, test2: list[str]) -> str:
            return f"{message}: {test1}, {", ".join(test2)}"


        agent = Agent(role="demo", goal="amazing project goal", maxit=1)

        task = Task(
            description="Amazing task",
            pydantic_output=CustomOutput,
            callback=dummy_func,
            callback_kwargs=dict(message="Hi! Here is the result: ")
        )

        res = task.execute_sync(agent=agent, context="amazing context to consider.")
        assert "Hi! Here is the result:" in res.callback_output
        assert [getattr(res.pydantic, k) for k, v in CustomOutput.model_fields.items()]

    a()
    b()


def test_doc_agent():
    """core/agent.md"""

    def a():
        from versionhq import Agent
        agent = Agent(
            role="Marketing Analyst",
            goal="Coping with price competition in saturated markets"
        )

        assert agent.id and agent.role == "Marketing Analyst" and agent.goal == "Coping with price competition in saturated markets" and agent.backstory is not None

    def b():
        from versionhq import Agent
        agent = Agent(
            role="Marketing Analyst",
            goal="Coping with price competition in saturated markets",
            llm="gemini-2.0",
        )

        from versionhq.llm.model import LLM
        assert "gemini-2.0" in agent.llm.model and isinstance(agent.llm, LLM)


    def c_1():
        from versionhq import Agent
        agent = Agent(
            role="Marketing Analyst",
            goal="Coping with price competition in saturated markets"
        )

        assert agent.backstory == "You are an expert marketing analyst with relevant skillsets and abilities to query relevant information from the given knowledge sources. Leveraging these, you will identify competitive solutions to achieve the following goal: coping with price competition in saturated markets."


    def c_2():
        from versionhq import Agent

        agent = Agent(
            role="Marketing Analyst",
            goal="Coping with increased price competition in saturated markets.",
            backstory="You are a marketing analyst for a company in a saturated market. The market is becoming increasingly price-competitive, and your company's profit margins are shrinking. Your primary goal is to develop and implement strategies to help your company maintain its market share and profitability in this challenging environment."
        )
        assert agent.backstory == "You are a marketing analyst for a company in a saturated market. The market is becoming increasingly price-competitive, and your company's profit margins are shrinking. Your primary goal is to develop and implement strategies to help your company maintain its market share and profitability in this challenging environment."


    def e_1():
        import json
        from typing import Dict, Any
        from versionhq import Agent

        def format_response(res: str = None) -> str | Dict[str, Any]:
            try:
                r = json.dumps(eval(res))
                formatted_res = json.loads(r)
                return formatted_res
            except:
                return res

        agent = Agent(
            role="Marketing Analyst",
            goal="Coping with increased price competition in saturated markets.",
            callbacks=[format_response]
        )

        import litellm
        assert litellm.callbacks == agent.callbacks


    def e_2():
        import json
        from typing import Dict, Any
        from versionhq import Agent, Task

        def assessment(res: str) -> str:
            try:
                sub_agent = Agent(role="Validator", goal="Validate the given solutions.")
                task = Task(description=f"""Assess the given solution based on feasibilities and fits to client's strategies, then refine the solution if necessary.
        Solution: {res}
        """)
                r = task.sync_execute(agent=sub_agent)
                return r.raw

            except:
                return res

        def format_response(res: str = None) -> str | Dict[str, Any]:
            try:
                r = json.dumps(eval(res))
                formatted_res = json.loads(r)
                return formatted_res
            except:
                return res

        agent = Agent(
            role="Marketing Analyst",
            goal="Build solutions to address increased price competition in saturated markets",
            callbacks=[assessment, format_response]
        )

        import litellm
        assert litellm.callbacks == agent.callbacks and len(agent.callbacks) == 2


    def f():
        from versionhq import Agent

        llm_config = dict(
            temperature=1,
            top_p=0.1,
            n=1,
            stream=False,
            stream_options=None,
            stop="test",
            max_completion_tokens=10000,
            dummy="I am dummy"
        )

        agent = Agent(
            role="Marketing Analyst",
            goal="Coping with increased price competition in saturated markets.",
            respect_context_window=False,
            max_tokens=3000,
            max_execution_time=60,
            max_rpm=5,
            llm_config=llm_config
        )

        assert agent.llm.max_tokens == 3000
        assert agent.llm.temperature == 1
        assert agent.llm.top_p == 0.1
        assert agent.llm.n == 1
        assert agent.llm.stop=="test"


    def g():
        from versionhq import Agent
        from versionhq.task.model import Task
        from versionhq.knowledge.source import StringKnowledgeSource

        content = "Kuriko's favorite color is gold, and she enjoy Japanese food."
        string_source = StringKnowledgeSource(content=content)

        agent = Agent(
            role="Information Agent",
            goal="Provide information based on knowledge sources",
            knowledge_sources=[string_source,]
        )

        task = Task(
            description="Answer the following question: What is Kuriko's favorite color?"
        )

        res = task.execute_sync(agent=agent)
        assert "gold" in res.raw.lower()


    def h():
        from versionhq import Agent

        agent = Agent(
            role="Researcher",
            goal="You research about math.",
            use_memory=True
        )

        from versionhq.memory.model import ShortTermMemory, LongTermMemory
        assert isinstance(agent.short_term_memory, ShortTermMemory) and isinstance(agent.long_term_memory, LongTermMemory)


    def z():
        from versionhq import Agent

        agent = Agent(
            config=dict(
                role="Marketing Analyst",
                goal="Coping with increased price competition in saturated markets.",
            )
        )

        assert agent.role == "Marketing Analyst" and agent.goal == "Coping with increased price competition in saturated markets."

    a()
    b()
    c_1()
    c_2()
    e_1()
    e_2()
    f()
    g()
    h()
    z()



"""core/tool.md"""

def test_doc_core_tool_a():
    from versionhq import Tool

    def demo_func(message: str) -> str:
        return message + "_demo"

    my_tool = Tool(func=demo_func)
    res = my_tool.run(params=dict(message="Hi!"))

    assert res == "Hi!_demo"


def test_doc_core_tool_b():
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
    assert agent.function_calling_llm.model is not None


def test_doc_core_tool_c():
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
    assert agent.function_calling_llm.model is not None


def test_doc_core_tool_d():
    from versionhq import Tool

    my_tool = Tool(name="my empty tool", func = lambda x: x)
    assert my_tool.name == "my empty tool" and my_tool.func is not None


    def demo_func() -> str:
        return "demo"

    my_tool_2 = Tool(func=demo_func)
    assert my_tool_2.name == "demo_func" and my_tool_2.func == demo_func


def test_doc_core_tool_e1():
    from versionhq import Tool

    my_tool = Tool(func=lambda x: f"demo-{x}" )
    res = my_tool._run(x="TESTING")
    assert res == "demo-TESTING"


def test_doc_core_tool_e2():
    from versionhq import Tool

    def demo_func() -> str:
        """...some complex execution..."""
        return "demo"

    my_tool = Tool(func=demo_func)
    res = my_tool._run()
    assert res == "demo"


def test_doc_core_tool_e3():
    from versionhq import Tool

    def demo_func(message: str) -> str:
        return message + "_demo"

    my_tool = Tool(func=demo_func)
    res = my_tool._run(message="Hi!")

    assert res == "Hi!_demo"


def test_doc_core_tool_f():
    from typing import Callable
    from versionhq import Tool

    class MyCustomTool(Tool):
        name: str = "custom tool"
        func: Callable

    my_custom_tool = MyCustomTool(func=lambda x: len(x))
    res = my_custom_tool._run(["demo1", "demo2"])
    assert res == 2


def test_doc_core_tool_g():
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


def test_doc_core_tool_h1():
    from versionhq import Tool, Agent, Task

    my_tool = Tool(name="demo tool", func=lambda x: "demo func")
    assert my_tool

    agent = Agent(
        role="demo",
        goal="execute tools",
        function_calling_llm="gpt-4o",
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

    res = task.execute_sync(agent=agent)
    assert res.raw == "demo func"


def test_doc_core_tool_h2():
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

    res = task.execute_sync(agent=agent)
    assert res.raw == "demo func" and res.tool_output == "demo func"


def test_doc_core_tool_h3():
    from versionhq import Tool, Agent, Task

    my_tool = Tool(name="demo tool", func=lambda x: "demo func")

    agent = Agent(
        role="Demo Tool Handler",
        goal="execute tools",
        tools=[my_tool],
        maxit=1,
        max_tokens=3000
    )

    task = Task(
        description="execute tools",
        can_use_agent_tools=True,
        tool_res_as_final=True
    )

    res = task.execute_sync(agent=agent)

    assert res.tool_output == "demo func"
    assert "Demo Tool Handler" in task.processed_agents


def test_doc_core_tool_h4():
    from versionhq.llm.model import LLM

    llm = LLM(model="gpt_4o")
    assert isinstance(llm._supports_function_calling(), bool)


def test_doc_core_tool_h5():
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
        goal="execute tools",
        maxit=1,
        max_tokens=3000
    )

    task = Task(
        description="execute the function",
        tools=[tool_set,], # use ToolSet to call args
        tool_res_as_final=True
    )

    res = task.execute_sync(agent=agent)
    assert res.tool_output == "empty func_demo"


def test_doc_core_tool_i():
    from versionhq.tool.decorator import tool

    @tool("demo")
    def my_tool(test_words: str) -> str:
        """Test a tool decorator."""
        return test_words

    assert my_tool.name == "demo"
    assert "Tool: demo" in my_tool.description and "'test_words': {'description': '', 'type': 'str'" in my_tool.description
    assert my_tool.func("testing") == "testing"
