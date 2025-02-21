from unittest.mock import patch
from typing import Callable

from versionhq.agent.model import Agent, LLM
from versionhq.task.model import Task, ResponseField, TaskOutput
from versionhq.tool.model import Tool, ToolSet
from versionhq.tool.decorator import tool


# def test_async_execute_task():
#     task = Task(description="Return string: 'test'", type=TaskExecutionType.ASYNC)

#     with patch.object(Agent, "execute_task", return_value="test") as execute:
#         res = task.execute()
#         assert res.raw == "test"
#         execute.assert_called_once_with(task=task, context=None, task_tools=list())


# def test_sync_execute_with_task_context():
#     """
#     Use case = One agent handling multiple tasks sequentially using context set in the main task.
#     """
#     sub_task = Task(
#         description="return the output following the given prompt.",
#         response_fields=[
#             ResponseField(title="subtask_result", data_type=str, required=True),
#         ]
#     )
#     main_task = Task(
#         description="return the output following the given prompt.",
#         response_fields=[
#             ResponseField(title="test1", data_type=int, required=True),
#             ResponseField(title="test2", data_type=str, required=True),
#         ],
#     )
#     res = main_task.execute(context=[sub_task])

#     assert isinstance(res, TaskOutput)
#     assert res.task_id is main_task.id
#     assert res.raw is not None
#     assert isinstance(res.raw, str)
#     assert res.json_dict is not None
#     assert isinstance(res.json_dict, dict)
#     assert res.pydantic is None
#     assert sub_task.output is not None
#     assert sub_task.output.json_dict is not None
#     assert "subtask_result" in main_task._prompt()



# def test_callback():
#     """
#     See if the callback function is executed well with kwargs.
#     """

#     def callback_func(condition: str, test1: str):
#         # task_id = str(id) if id else None
#         return f"Result: {test1}, condition added: {condition}"

#     task = Task(
#         description="return the output following the given prompt.",
#         response_fields=[
#             ResponseField(title="test1", data_type=str, required=True),
#         ],
#         callback=callback_func,
#         callback_kwargs=dict(condition="demo for pytest")
#     )
#     res = task.execute()

#     assert res and isinstance(res, TaskOutput)
#     assert res.task_id is task.id
#     assert "demo for pytest" in res.callback_output


# def test_delegate():
#     # agent = Agent(role="demo agent 6", goal="My amazing goals", maxit=1, max_tokens=3000)
#     task = Task(
#         description="return the output following the given prompt.",
#         response_fields=[
#             ResponseField(title="test1", data_type=str, required=True),
#         ],
#         allow_delegation=True
#     )
#     task.execute()

#     assert task.output is not None
#     assert "vhq-Delegated-Agent" in task.processed_agents
#     assert task.delegations != 0


# def test_conditional_task():
#     task = Task(
#         description="erturn the output following the given prompt.",
#         response_fields=[ResponseField(title="test1", data_type=str, required=True),],
#     )
#     res = task.execute_sync(agent=base_agent)

#     conditional_task = ConditionalTask(
#         description="return the output following the given prompt.",
#         response_fields=[ResponseField(title="test1", data_type=str, required=True),],
#         condition=lambda x: bool("zzz" in task.output.raw)
#     )
#     should_execute = conditional_task.should_execute(context=res)

#     assert res.raw is not None
#     assert should_execute is False

#     conditional_res = conditional_task._handle_conditional_task(task_outputs=[res,], task_index=1, was_replayed=False)
#     if not should_execute:
#         assert conditional_res is None
#     else:
#         assert conditional_res.task_id is conditional_task.id


def test_store_task_log():
    task = Task(
        description="return the output following the given prompt.",
        response_fields=[ResponseField(title="task_1", data_type=str, required=True),],
    )
    task.execute()

    from versionhq.storage.task_output_storage import TaskOutputStorageHandler
    assert TaskOutputStorageHandler().load() is not None


def test_task_with_agent_tools():
    simple_tool = Tool(name="simple tool", func=lambda x: "simple func")
    agent = Agent(role="demo", goal="execute tools", tools=[simple_tool,], maxit=1, max_tokens=3000)
    task = Task(description="execute tool", can_use_agent_tools=True, tool_res_as_final=True)
    res = task.execute(agent=agent)
    assert res.tool_output == "simple func"

    def empty_func():
        return "empty func"

    func_tool = Tool(name="func tool", func=empty_func)
    agent.tools = [func_tool]
    res = task.execute(agent=agent)
    assert res.tool_output == "empty func"


    def demo_func(message: str) -> str:
        return message + "_demo"

    class CustomTool(Tool):
        name: str = "custom tool"

    custom_tool = CustomTool(func=demo_func)
    agent.tools = [custom_tool]
    res = task.execute(agent=agent)
    assert "_demo" in res.tool_output


def test_task_with_tools():

    def random_func(message: str) -> str:
        return message + "_demo"

    tool = Tool(name="tool", func=random_func)
    tool_set = ToolSet(tool=tool, kwargs=dict(message="empty func"))
    task = Task(description="execute the given tools", tools=[tool_set,], tool_res_as_final=True)
    # res = task.execute()
    # assert res.tool_output == "empty func_demo"

    class CustomTool(Tool):
        name: str = "custom tool"
        func: Callable = None

    custom_tool = CustomTool(func=random_func)
    task.tools = [custom_tool]
    res = task.execute()
    assert "_demo" in res.tool_output

    task.tools = [custom_tool]
    res = task.execute()
    assert res.tool_output is not None



def test_task_without_response_format():
    task = Task(description="return a simple output with any random values.")
    res = task.execute()

    assert res and isinstance(res, TaskOutput)
    assert res.json_dict and isinstance(res.json_dict, dict)
    assert res.pydantic is None


def test_build_agent_without_developer_prompt():
    agent = Agent(
        role="analyst",
        goal="analyze the company's website and retrieve the product overview",
        backstory="You are competitive analysts who have abundand knowledge in marketing, product management.",
        use_developer_prompt=False
    )
    task = Task(description="return a simple output with any random values.")
    res = task.execute(agent=agent)

    assert res and isinstance(res, TaskOutput)
    assert res.json_dict and isinstance(res.json_dict, dict)
    assert res.pydantic is None


# def test_callback_with_custom_output():
#     class CustomOutput(BaseModel):
#         test1: str
#         test2: list[str]

#     def dummy_func(message: str, test1: str, test2: list[str]) -> str:
#         return f"""{message}: {test1}, {", ".join(test2)}"""

#     task = Task(
#         description="Amazing task",
#         pydantic_output=CustomOutput,
#         callback=dummy_func,
#         callback_kwargs=dict(message="Hi! Here is the result: ")
#     )
#     res = task.execute(context="amazing context to consider.")

#     assert res.task_id == task.id
#     assert res.pydantic.test1 and res.pydantic.test2
#     assert "Hi! Here is the result: " in res.callback_output and res.pydantic.test1 in res.callback_output and ", ".join(res.pydantic.test2) in res.callback_output


def test_task_with_agent_callback():
    import litellm

    def dummy_func(*args, **kwargs) -> str:
        return "Demo func"

    agent = Agent(role="demo", goal="amazing project goal", maxit=1, max_tokens=3000, callbacks=[dummy_func,])
    task = Task(description="Amazing task")
    res = task.execute(agent=agent)

    assert res.raw and res.task_id == task.id
    assert litellm.callbacks == [dummy_func]


def test_rpm():
    agent = Agent(role="demo", goal="use the given tools", max_tokens=3000, max_rpm=3)
    assert agent._rpm_controller and agent._rpm_controller.max_rpm == 3

    a = "hello"
    tool = Tool(func=lambda x: a + x)
    tool_set = ToolSet(tool=tool, kwargs={ "x": "_demo" })
    task = Task(description="Summarize overview of the given tool in sentence, then execute the tool.", tools=[tool_set,])
    res = task.execute(agent=agent)

    assert "hello_demo" in res.raw if res.raw else res
    assert agent._rpm_controller._current_rpm < agent._rpm_controller.max_rpm


def test_maxit():
    @tool
    def demo() -> str:
        """Get the final answer but don't give it yet, just re-use this
        tool non-stop."""
        return "demo"

    agent = Agent(role="demo", goal="amazing demo", maxit=2)
    task = Task(description="Summarize overview of the given tool in sentences.", tools=[demo,])

    with patch.object(LLM, "call", wraps=agent.llm.call) as mock:
        task.execute(agent=agent)
        assert mock.call_count <= 2
