from typing import Dict, Any

from pydantic import BaseModel

from versionhq.agent.model import Agent
from versionhq.task.model import ResponseField
from versionhq.llm.model import DEFAULT_MODEL_NAME, LLM


class DemoChild(BaseModel):
    """
    A nested outcome class.
    """
    ch_1: str
    ch_2: dict[str, str]


class DemoOutcome(BaseModel):
    """
    A demo pydantic class to validate the outcome with various nested data types.
    """
    test0: int
    test1: float
    test2: str
    test3: bool
    test4: list[str]
    test5: dict[str, Any]
    test6: list[dict[str, Any]]
    test8: list[list[str]]
    # children: List[DemoChild]


demo_response_fields = [
    ResponseField(title="test0", data_type=int),
    ResponseField(title="test1", data_type=str, required=True),
    ResponseField(title="test2", data_type=list, items=str),
    ResponseField(title="test3", data_type=list, items=dict, properties=[
        ResponseField(title="nest1", data_type=str),
        ResponseField(title="nest2", type=dict, properties=[ResponseField(title="test", data_type=str)])
    ]),
    ResponseField(title="test4", data_type=dict, properties=[ResponseField(title="ch", data_type=tuple)]),
    ResponseField(title="test5", data_type=bool),
    ResponseField(title="test6", data_type=list, items=Any, required=False),
    # ResponseField(title="children", data_type=list, items=type(DemoChild)),
]


def create_base_agent(model: str | LLM | Dict[str, Any]) -> Agent:
    agent = Agent(role="demo", goal="My amazing goals", llm=model, max_tokens=3000, maxit=1)
    return agent


base_agent = create_base_agent(model=DEFAULT_MODEL_NAME)
