from unittest.mock import patch

from versionhq._utils import UsageMetrics
from versionhq._prompt.model import Prompt


def test_gpt_cup():
    import versionhq as vhq

    params = dict(
        user_prompt="test",
        img_url="/sample.com",
        reasoning_effort="medium",
        tools=vhq.CUPToolSchema(display_width=500, display_height=300, environment="mac", type="computer_use_preview")
    )
    tool = vhq.GPTToolCUP(**params)

    assert tool.user_prompt == params["user_prompt"]
    assert tool.img_url is None
    assert tool.reasoning_effort == "medium"
    assert isinstance(tool.tools, list)
    assert tool.tools[0].display_width == 500
    assert tool.tools[0].environment == "mac"
    assert tool.tools[0].type == "computer_use_preview"
    assert tool.schema is not None

    raw, _, usage = tool.run()
    assert raw is not None if usage.total_errors == 0 else raw is ""
    assert isinstance(usage, UsageMetrics)
    assert usage.total_tokens is not None


def test_gpt_web_search():
    import versionhq as vhq

    params = dict(input="Search today's top news.", country="US", city="test", search_content_size = "high")
    tool = vhq.GPTToolWebSearch(**params)
    assert tool.model == "gpt-4o"
    assert tool.input == params["input"]
    assert tool._user_location == None
    assert tool.search_content_size == "high"
    assert tool.schema is not None

    raw, annotations, usage = tool.run()
    assert raw is not None if usage.total_errors == 0 else raw is ""
    assert isinstance(annotations, list)
    assert isinstance(usage, UsageMetrics)
    assert usage.total_tokens is not None


def test_gpt_file_search():
    import versionhq as vhq

    params = dict(
        input="Search today's top news.",
        vector_store_ids="vs_dummy_id",
        max_num_results=5,
    )
    tool = vhq.GPTToolFileSearch(**params)
    assert tool.model == "gpt-4o"
    assert tool.input == params["input"]
    assert tool.vector_store_ids == [params["vector_store_ids"]]
    assert tool.max_num_results == params["max_num_results"]
    assert tool.schema is not None

    with patch.object(vhq.GPTToolFileSearch, "run", return_value=("test", [], UsageMetrics())) as mock_run:
        tool.run()
        mock_run.assert_called_once()


def test_with_agent():
    import versionhq as vhq

    tool_1 = vhq.GPTToolWebSearch(input="Search today's top news.", search_content_size = "high")
    tool_2 = vhq.GPTToolFileSearch(input="Search today's top news.", vector_store_ids="vs_dummy_id", max_num_results=5)

    agent = vhq.Agent(role="GPT Tool Handling", tools=[tool_1])
    with patch.object(vhq.Agent, "_handle_gpt_tools", return_value=(vhq.TaskOutput(raw=""))) as mock_run:
        agent.start(tool_res_as_final=True)
        mock_run.assert_called_once()

    with patch.object(Prompt, "format_core", return_value=("test", "test", ["test"])) as mock_prompt:
        agent.start()
        mock_prompt.assert_called_once()


    agent = vhq.Agent(role="GPT Tool Handling", tools=[tool_2])
    with patch.object(vhq.Agent, "_handle_gpt_tools", return_value=(vhq.TaskOutput(raw=""))) as mock_run:
        agent.start(tool_res_as_final=True)
        mock_run.assert_called_once()

    with patch.object(Prompt, "format_core", return_value=("test", "test", ["test"])) as mock_prompt:
        agent.start()
        mock_prompt.assert_called_once()


    agent = vhq.Agent(role="GPT Tool Handling", tools=[tool_1, tool_2])
    with patch.object(vhq.Agent, "_handle_gpt_tools", return_value=("test", "test", vhq.TaskOutput(raw=""), UsageMetrics())) as mock_run:
        agent.start(tool_res_as_final=True)
        mock_run.assert_called_once()

    with patch.object(Prompt, "format_core", return_value=("test", "test", ["test"])) as mock_prompt:
        agent.start()
        mock_prompt.assert_called_once()


def test_with_task():
    import versionhq as vhq

    tool_1 = vhq.GPTToolWebSearch(input="Search today's top news.", search_content_size = "high")
    tool_2 = vhq.GPTToolFileSearch(input="Search today's top news.", vector_store_ids="vs_dummy_id", max_num_results=5)

    task = vhq.Task(description="Test gpt tools", tools=[tool_1], tool_res_as_final=True)
    with patch.object(vhq.Agent, "_handle_gpt_tools", return_value=(vhq.TaskOutput(raw="", usage=UsageMetrics()))) as mock_run:
        task.execute()
        mock_run.assert_called_once()

    with patch.object(Prompt, "format_core", return_value=("test", "test", ["test"])) as mock_prompt:
        task.tool_res_as_final = False
        task.execute()
        mock_prompt.assert_called_once()


    task = vhq.Task(description="Test gpt tools", tools=[tool_2], tool_res_as_final=True)
    with patch.object(vhq.Agent, "_handle_gpt_tools", return_value=(vhq.TaskOutput(raw="", usage=UsageMetrics()))) as mock_run:
        task.execute()
        mock_run.assert_called_once()

    with patch.object(Prompt, "format_core", return_value=("test", "test", ["test"])) as mock_prompt:
        task.tool_res_as_final = False
        task.execute()
        mock_prompt.assert_called_once()


    task = vhq.Task(description="Test gpt tools", tools=[tool_1, tool_2], tool_res_as_final=True)
    with patch.object(vhq.Agent, "_handle_gpt_tools", return_value=(vhq.TaskOutput(raw="", usage=UsageMetrics()))) as mock_run:
        task.execute()
        mock_run.assert_called_once()

    with patch.object(Prompt, "format_core", return_value=("test", "test", ["test"])) as mock_prompt:
        task.tool_res_as_final = False
        task.execute()
        mock_prompt.assert_called_once()
