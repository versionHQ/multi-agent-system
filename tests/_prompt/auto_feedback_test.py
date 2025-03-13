from pydantic import BaseModel

from versionhq._prompt.model import Prompt
from versionhq.task_graph.model import ReformTriggerEvent
from versionhq._prompt.auto_feedback import PromptFeedbackGraph
from versionhq.agent.model import Agent
from versionhq.task.model import Task, TaskOutput


def test_pfg():
    class Custom(BaseModel):
        schedule: str
        destination: str
        other_consideration: str

    main_task = Task(description="plan a day trip to Costa Rica", pydantic_output=Custom)
    agent = Agent(llm="gemini-2.0", role="Demo Agent")
    prompt  = Prompt(task=main_task, agent=agent, context=["test", "test2"])
    pfg = PromptFeedbackGraph(prompt=prompt)
    assert pfg.prompts["0"] == prompt
    assert pfg._times_iteration == 0
    assert pfg.user_prompts["0"] is not None
    assert pfg.dev_prompts["0"] is not None

    pfg.set_up_graph(should_reform=False, reform_trigger_event=ReformTriggerEvent.ERROR_DETECTION)
    assert len([k for k in pfg.nodes.keys()]) == 5
    assert pfg.edges is not None
    assert pfg.should_reform == False
    assert pfg.reform_trigger_event == ReformTriggerEvent.ERROR_DETECTION

    res, all_outputs = pfg.activate()
    assert isinstance(res, TaskOutput)
    assert res.pydantic.schedule and res.pydantic.destination and res.pydantic.other_consideration
    assert all_outputs
