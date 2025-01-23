import uuid
from abc import ABC
from datetime import date, datetime, time, timedelta
from typing import Any, Dict, List, Callable, Type, Optional, get_args, get_origin
from typing_extensions import Self
from pydantic import UUID4, InstanceOf, BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic_core import PydanticCustomError

from versionhq.clients.product.model import Product
from versionhq.clients.customer.model import Customer
from versionhq.agent.model import Agent
from versionhq.team.model import Team
from versionhq.tool.composio_tool_vars import ComposioAppName


class ScoreFormat:
    def __init__(self, rate: float | int = 0, weight: int = 1):
        self.rate = rate
        self.weight = weight
        self.aggregate = rate * weight


class Score:
    """
    Evaluate the score on 0 (no performance) to 1 scale.
    `rate`: Any float from 0.0 to 1.0 given by an agent.
    `weight`: Importance of each factor to the aggregated score.
    """

    def __init__(
        self,
        brand_tone: ScoreFormat = ScoreFormat(0, 0),
        audience: ScoreFormat = ScoreFormat(0, 0),
        track_record: ScoreFormat = ScoreFormat(0, 0),
        **kwargs: Optional[Dict[str, ScoreFormat]],
    ):
        self.brand_tone = brand_tone
        self.audience = audience
        self.track_record = track_record
        self.kwargs = kwargs


    def result(self) -> int:
        aggregate_score = int(self.brand_tone.aggregate) + int(self.audience.aggregate) + int(self.track_record.aggregate)
        denominator = self.brand_tone.weight + self.audience.weight + self.track_record.weight

        for k, v in self.kwargs.items():
            aggregate_score += v.aggregate
            denominator += v.weight

        if denominator == 0:
            return 0

        return round(aggregate_score / denominator, 2)



class MessagingComponent(ABC, BaseModel):
    layer_id: int = Field(default=0, description="add id of the layer: 0, 1, 2")
    message: str = Field(default=None, max_length=1024, description="text message content to be sent")
    score: InstanceOf[Score] = Field(default=None)
    condition: str = Field(default=None, description="condition to execute the next component")
    interval: Optional[str] = Field(default=None, description="ideal interval to set to assess the condition")


    def store_scoring_result(self, subject: str, score_raw: int | Score | ScoreFormat = None) -> Self:
        """
        Set up the `score` field
        """

        if isinstance(score_raw, Score):
            setattr(self, "score", score_raw)

        elif isinstance(score_raw, ScoreFormat):
            score_instance = Score()
            setattr(score_instance, subject, score_raw)
            setattr(self, "score", score_instance)

        elif isinstance(score_raw, int) or isinstance(score_raw, float):
            score_instance, score_format_instance = Score(), ScoreFormat(rate=score_raw, weight=1)
            setattr(score_instance, "kwargs", { subject: score_format_instance })
            setattr(self, "score", score_instance)

        else:
            pass

        return self



class MessagingWorkflow(ABC, BaseModel):
    """
    Store 3 layers of messaging workflow sent to `customer` on the `product`
    """

    _created_at: Optional[datetime]
    _updated_at: Optional[datetime]

    model_config = ConfigDict(extra="allow")

    id: UUID4 = Field(default_factory=uuid.uuid4, frozen=True)
    messaging_components: List[MessagingComponent] = Field(default_factory=list, description="store messaging components in the workflow")

    # responsible tean or agents
    team: Optional[Team] = Field(default=None, description="store a responsibile team to autopilot the workflow")
    agents: Optional[List[Agent]] = Field(default=None, description="store responsible agents. None when the team exists")

    # metrics
    destination: Optional[ComposioAppName | str] = Field(default=None, description="destination service to launch the workflow")
    product: InstanceOf[Product] = Field(default=None)
    customer: InstanceOf[Customer] = Field(default=None)
    performance_metrics: List[Dict[str, Any]] | List[str] = Field(default=None, max_length=256, description="performance metrics to track")

    @field_validator("id", mode="before")
    @classmethod
    def _deny_user_set_id(cls, v: Optional[UUID4]) -> None:
        if v:
            raise PydanticCustomError("may_not_set_field", "This field is not to be set by the user.", {})


    @model_validator(mode="after")
    def set_up_destination(self):
        """
        Set up the destination service using ComposioAppName class.
        """
        if isinstance(self.destination, ComposioAppName):
            pass

        elif isinstance(self.destination, str) and self.destination in ComposioAppName:
            self.destination = ComposioAppName(self.destination)

        elif self.destination is None:
            # if self.customer is not None:
            #     self.destination = self.customer.on

            if self.product.provider is not None and self.product.provider.destination_services:
                applied_service = self.product.provider.destination_services[0]
                self.destination = ComposioAppName(applied_service) if applied_service in ComposioAppName else applied_service

        return self


    def reassign_agent_or_team(self, agents: List[Agent] = None, team: Team = None) -> None:
        """
        Fire unresponsible agents/team and assign new one.
        """

        if not agents and not team:
            raise ValueError("Need to add at least 1 agent or team.")

        self.agents = agents
        self.team = team
        self.updated_at = datetime.datetime.now()


    @property
    def name(self) -> str:
        if self.customer.id:
            return f"Workflow ID: {self.id} - on {self.product.id} for {self.customer.id}"
        else:
            return f"Workflow ID: {self.id} - on {self.product.id}"
