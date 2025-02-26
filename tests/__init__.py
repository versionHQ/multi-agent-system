from typing import Any
from pydantic import BaseModel
import versionhq as vhq


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
    demo_nest_1: list[dict[str, Any]]
    demo_nest_2: list[list[str]]
    demo_nest_3: dict[str, list[str]]
    demo_nest_4: dict[str, dict[str, Any]]


demo_response_fields = [
    vhq.ResponseField(title="demo_1", data_type=int),
    vhq.ResponseField(title="demo_2", data_type=float),
    vhq.ResponseField(title="demo_3", data_type=str),
    vhq.ResponseField(title="demo_4", data_type=bool),
    vhq.ResponseField(title="demo_5", data_type=list, items=str),
    vhq.ResponseField(title="demo_6", data_type=dict, properties=[vhq.ResponseField(title="demo-item", data_type=str)]),
    vhq.ResponseField(title="demo_nest_1", data_type=list, items=dict, properties=([
        vhq.ResponseField(title="nest1", data_type=dict, properties=[vhq.ResponseField(title="nest11", data_type=str)])
    ])),
    vhq.ResponseField(title="demo_nest_2", data_type=list, items=list),
    vhq.ResponseField(title="demo_nest_3", data_type=dict, properties=[
        vhq.ResponseField(title="nest1", data_type=list, items=str)
    ]),
    vhq.ResponseField(title="demo_nest_4", data_type=dict, properties=[
        vhq.ResponseField(title="nest1", data_type=dict, properties=[vhq.ResponseField(title="nest12", data_type=str)])
    ])
]
