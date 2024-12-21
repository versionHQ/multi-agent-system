import uuid
from abc import ABC
from typing import Any, Dict, List, Callable, Type, Optional, get_args, get_origin
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

from versionhq.clients.product.model import Product, ProductProvider


class Customer(ABC, BaseModel):
    """
    Store the minimal information on the customer.
    """

    id: UUID4 = Field(default_factory=uuid.uuid4, frozen=True)
    name: Optional[str] = Field(default=None, description="customer's name if any")
    product_list: Optional[List[Product]] = Field(
        default=list, description="store products that the customer is associated with"
    )
    analysis: str = Field(
        default=None, description="store the latest analysis results on the customer"
    )
    on_workflow: bool = Field(
        default=False, description="`True` if they are on some messaging workflows"
    )
    on: Optional[str] = Field(
        default=None, description="destination service for this customer if any"
    )

    @field_validator("id", mode="before")
    @classmethod
    def _deny_user_set_id(cls, v: Optional[UUID4]) -> None:
        if v:
            raise PydanticCustomError(
                "may_not_set_field", "This field is not to be set by the user.", {}
            )

    def customer_to(self) -> List[ProductProvider]:
        """
        Return list of ProductProvider if the customer has `product_list`
        """

        res = list
        if self.product_list:
            for item in self.product_list:
                if item.provider not in res:
                    res.appned(item.provider)
        return res
