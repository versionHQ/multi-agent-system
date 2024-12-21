import uuid
from abc import ABC
from datetime import date, datetime, time, timedelta
from typing import (
    Any,
    Dict,
    List,
    Union,
    Callable,
    Type,
    Optional,
    get_args,
    get_origin,
)
from pydantic import (
    UUID4,
    InstanceOf,
    BaseModel,
    ConfigDict,
    Field,
    create_model,
    field_validator,
    model_validator,
)
from pydantic_core import PydanticCustomError

from versionhq.clients.product.model import Product
from versionhq.clients.customer.model import Customer
from versionhq.agent.model import Agent
from versionhq.team.model import Team


class ScoreFormat:
    def __init__(self, rate: float, weight: int = 1):
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
        brand_tone: ScoreFormat,
        audience: ScoreFormat,
        track_record: ScoreFormat,
        *args: List[ScoreFormat],
    ):
        self.brand_tone = brand_tone
        self.audience = audience
        self.track_record = track_record
        self.args = args

    def result(self):
        aggregated_score = sum(
            self.brand_tone.aggregate,
            self.audience.aggregate,
            self.track_record.aggrigate,
        )
        denominator = sum(
            self.brand_tone.weight, self.audience.weight, self.track_record.weight
        )
        try:
            if self.args:
                for item in self.args:
                    if isinstance(item, ScoreFormat):
                        aggregate_score += item.rate * item.weight
                        denominator += item.weight
        except:
            pass
        return round(aggregated_score / denominator, 2)


class MessagingComponent(ABC, BaseModel):
    layer_id: int = Field(default=0, description="add id of the layer: 0, 1, 2")
    message: str = Field(
        default=None, max_length=1024, description="text message content to be sent"
    )
    interval: Optional[str] = Field(
        default=None,
        description="interval to move on to the next layer. if this is the last layer, set as `None`",
    )
    score: Union[float, InstanceOf[Score]] = Field(default=None)


class MessagingWorkflow(ABC, BaseModel):
    """
    Store 3 layers of messaging workflow sent to `customer` on the `product`
    """

    _created_at: Optional[datetime]
    _updated_at: Optional[datetime]

    model_config = ConfigDict()

    id: UUID4 = Field(default_factory=uuid.uuid4, frozen=True)
    comps: List[MessagingComponent] = Field(
        default=list, description="store at least 3 messaging components"
    )

    # responsible tean or agents
    team: Optional[Team] = Field(
        default=None,
        description="store `Team` instance responsibile for autopiloting this workflow",
    )
    agents: Optional[List[Agent]] = Field(
        default=None,
        description="store `Agent` instances responsible for autopiloting this workflow. if the team exsits, this field remains as `None`",
    )

    # metrics
    destination: Optional[str] = Field(
        default=None, description="destination service to launch this workflow"
    )
    product: InstanceOf[Product] = Field(default=None)
    customer: InstanceOf[Customer] = Field(default=None)

    metrics: Union[List[Dict[str, Any]], List[str]] = Field(
        default=None,
        max_length=256,
        description="store metrics that used to predict and track the performance of this workflow.",
    )

    @property
    def name(self):
        if self.customer.id:
            return (
                f"Workflow ID: {self.id} - on {self.product.id} for {self.customer.id}"
            )
        else:
            return f"Workflow ID: {self.id} - on {self.product.id}"

    @field_validator("id", mode="before")
    @classmethod
    def _deny_user_set_id(cls, v: Optional[UUID4]) -> None:
        if v:
            raise PydanticCustomError(
                "may_not_set_field", "This field is not to be set by the user.", {}
            )

    @model_validator(mode="after")
    def set_up_destination(self):
        """
        Set up the destination service when self.destination is None.
        Prioritize customer's destination to the product provider's destination list.
        """
        if self.destination is None:
            if self.customer is not None:
                self.destination = self.customer.on

            else:
                destination_list = self.product.provider.destinations
                if destination_list:
                    self.destination = destination_list[0]
        return self

    def reassign_agent_or_team(
        self, agents: List[Agent] = None, team: Team = None
    ) -> None:
        """
        Fire unresponsible agents/team and assign new one.
        """

        if not agents and not team:
            raise ValueError("Need to add at least 1 agent or team.")

        self.agents = agents
        self.team = team
        self.updated_at = datetime.datetime.now()
