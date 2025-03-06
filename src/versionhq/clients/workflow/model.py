import uuid
from abc import ABC
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import UUID4, InstanceOf, BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic_core import PydanticCustomError

from versionhq.agent.model import Agent
from versionhq.agent_network.model import AgentNetwork
from versionhq.clients.product.model import Product
from versionhq.clients.customer.model import Customer
from versionhq.tool.composio_tool_vars import ComposioAppName


class MessagingComponent(ABC, BaseModel):
    layer_id: int = Field(default=0, description="add id of the layer: 0, 1, 2")
    message: str = Field(default=None, max_length=1024, description="text message content to be sent")
    score: Optional[float | int] = Field(default=None)
    condition: str = Field(default=None, description="condition to execute the next component")
    interval: Optional[str] = Field(default=None, description="ideal interval to set to assess the condition")


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
    agent_network: Optional[AgentNetwork] = Field(default=None, description="store a responsibile agent network to autopilot the workflow")
    agents: Optional[List[Agent]] = Field(default=None, description="store responsible agents. None when the `agent_network` fields has a value")

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


    def reassign_agent(self, agents: List[Agent] = None, agent_network: AgentNetwork = None) -> None:
        """
        Switch agents
        """

        if not agents and not agent_network:
            raise ValueError("Missing agent or agent network to assign.")

        self.agents = agents
        self.agent_network = agent_network
        self.updated_at = datetime.datetime.now()


    @property
    def name(self) -> str:
        if self.customer.id:
            return f"Workflow ID: {self.id} - on {self.product.id} for {self.customer.id}"
        else:
            return f"Workflow ID: {self.id} - on {self.product.id}"
