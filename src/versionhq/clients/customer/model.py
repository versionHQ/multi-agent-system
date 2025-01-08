import uuid
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Callable, Type, Optional, get_args, get_origin
from pydantic import UUID4, InstanceOf, BaseModel, ConfigDict, Field, create_model, field_validator, model_validator
from pydantic_core import PydanticCustomError

from versionhq.clients.product.model import Product, ProductProvider
from versionhq.clients.customer import Status


class BaseCustomer(ABC, BaseModel):
    """
    Abstract base class for the base customer storing current status on the workflow and deployment method.
    """
    status: Status = Field(default=Status.NOT_ASSIGNED)

    @abstractmethod
    def _deploy(self, *args, **kwargs) -> Any:
        """Any method to deploy targeting the customer"""


    def deploy(self, *args, **kwargs) -> Any:
        if self.status is Status.READY_TO_DEPLOY:
            return self._deploy(self, **args, **kwargs)



class Customer(BaseCustomer):
    """
    Customer class to store customer info and handle deployment methods.
    """

    id: UUID4 = Field(default_factory=uuid.uuid4, frozen=True)
    name: Optional[str] = Field(default=None, description="customer's name if any")
    products: Optional[List[Product]] = Field(default=list, description="store products that the customer is associated with")
    analysis: str = Field(default=None, description="store the latest analysis results on the customer")
    function: Optional[Callable] = Field(default=None, descripition="store deploy function")
    config: Optional[Dict[str, Any]] = Field(default=None, description="config to the function")


    @field_validator("id", mode="before")
    @classmethod
    def _deny_user_set_id(cls, v: Optional[UUID4]) -> None:
        if v:
            raise PydanticCustomError("may_not_set_field", "This field is not to be set by the user.", {})


    def fetch_product_providers(self) -> List[ProductProvider] | None:
        """
        Return list of ProductProvider if the customer has `product_list`
        """
        res = []
        if self.products:
            for item in self.products:
                if item.provider not in res:
                    res.appned(item.provider)
        return res

    def _deploy(self, *args, **kwargs):
        return self.deploy(self, *args, **kwargs)


    def deploy(self, *args, **kwargs):
        self.status = Status.ACTIVE_ON_WORKFLOW

        if self.function:
            return self.function(**self.config)

        return super().deploy(*args, **kwargs)
