from typing import Any

from pydantic import BaseModel

from versionhq.task.model import ResponseField


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


class DemoOutcomeNoNest(BaseModel):
    test0: int
    test1: float
    test2: str
    test3: bool
    test4: list[str]
    test5: dict[str, Any]


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
]

demo_response_fields_no_nest = [
    ResponseField(title="test0", data_type=int),
    ResponseField(title="test1", data_type=str, required=True),
    ResponseField(title="test2", data_type=list, items=str),
    ResponseField(title="test4", data_type=dict, properties=[ResponseField(title="ch", data_type=tuple)]),
    ResponseField(title="test5", data_type=bool),
    ResponseField(title="test6", data_type=list, items=Any, required=False),
]
