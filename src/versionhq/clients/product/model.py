import uuid
from typing import Any, Dict, List, Callable, Type, Optional, get_args, get_origin
from pydantic import UUID4, InstanceOf, BaseModel, ConfigDict, Field, create_model, field_validator, model_validator
from pydantic_core import PydanticCustomError


class ProductProvider(BaseModel):
    """
    Store the minimal client information.
    `data_pipeline` and `destinations` are for composio plug-in.
    (!REFINEME) Create an Enum list for the options.
    (!REFINEME) Create an Enum list for regions.
    """

    id: UUID4 = Field(default_factory=uuid.uuid4, frozen=True)
    name: Optional[str] = Field(default=None, description="client name")
    region: Optional[str] = Field(default=None, description="region of client's main business operation")
    data_pipeline: Optional[List[str]] = Field(default=None, description="store the data pipelines that the client is using")
    destinations: Optional[List[str]] = Field(default=None,description="store the destination services that the client is using")

    @field_validator("id", mode="before")
    @classmethod
    def _deny_user_set_id(cls, v: Optional[UUID4]) -> None:
        if v:
            raise PydanticCustomError("may_not_set_field", "This field is not to be set by the user.", {})



class Product(BaseModel):
    """
    Store the product information necessary to the outbound effrots and connect it to the `ProductProvider` instance.
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


    @field_validator("id", mode="before")
    @classmethod
    def _deny_user_set_id(cls, v: Optional[UUID4]) -> None:
        if v:
            raise PydanticCustomError("may_not_set_field", "This field is not to be set by the user.", {})
