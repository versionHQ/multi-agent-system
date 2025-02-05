import uuid
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Callable, Type, Optional, get_args, get_origin

from pydantic import UUID4, InstanceOf, BaseModel, ConfigDict, Field, create_model, field_validator, model_validator
from pydantic_core import PydanticCustomError

from versionhq.tool.composio_tool_vars import ComposioAppName


class ProductProvider(ABC, BaseModel):
    """
    Abstract class for the product provider entity.
    """

    id: UUID4 = Field(default_factory=uuid.uuid4, frozen=True)
    name: Optional[str] = Field(default=None)
    region: Optional[str] = Field(default=None, description="region of client's main business operation")
    data_pipelines: Optional[List[ComposioAppName | str]] = Field(default=None)
    destination_services: Optional[List[ComposioAppName | str]] = Field(default=None)

    @field_validator("id", mode="before")
    @classmethod
    def _deny_user_set_id(cls, v: Optional[UUID4]) -> None:
        if v:
            raise PydanticCustomError("may_not_set_field", "This field is not to be set by the user.", {})


    @model_validator(mode="after")
    def set_up_destinations(self):
        """
        Set up the destination services and data pipeines using ComposioAppName class.
        """
        if self.destination_services is not None:
            results = []
            for item in self.destination_services:
                if isinstance(item, ComposioAppName):
                    results.append(item)
                elif isinstance(item, str) and item in ComposioAppName:
                    results.append(ComposioAppName(item))
                else:
                    results.append(item)
            self.destination_services = results

        if self.data_pipelines is not None:
            results = []
            for item in self.data_pipelines:
                if isinstance(item, ComposioAppName):
                    results.append(item)
                elif isinstance(item, str) and item in ComposioAppName:
                    results.append(ComposioAppName(item))
                else:
                    results.append(item)
            self.data_pipelines = results

        return self


class Product(BaseModel):
    """
    A class to store product information used to create outbound
    """
    id: UUID4 = Field(default_factory=uuid.uuid4, frozen=True)
    name: Optional[str] = Field(default=None, description="product name")
    description: Optional[str] = Field(
        default=None,
        max_length=256,
        description="product description scraped from landing url or client input. cascade to the agent"
    )
    provider: Optional[ProductProvider] = Field(default=None)
    audience: Optional[str] = Field(default=None, description="target audience")
    usp: Optional[str] = Field(default=None)
    landing_url: Optional[str] = Field(default=None, description="marketing url of the product if any")
    cohort_timeframe: Optional[int] = Field(default=30, description="ideal cohort timeframe of the product in days")
    notes: Optional[str] = Field(default=None, description="any notes from the client to consider. cascade to the agent")


    @field_validator("id", mode="before")
    @classmethod
    def _deny_user_set_id(cls, v: Optional[UUID4]) -> None:
        if v:
            raise PydanticCustomError("may_not_set_field", "This field is not to be set by the user.", {})


    @field_validator("cohort_timeframe", mode="before")
    @classmethod
    def _deny_non_int_input(cls, v: Optional[UUID4]) -> None:
        if not isinstance(v, int):
            raise PydanticCustomError("invalid_input", "This field only accepts inputs in integer.", {})
