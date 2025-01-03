import os
import uuid
from dotenv import load_dotenv
from typing import Any, Callable, Type, get_args, get_origin, Optional, Tuple

from pydantic import BaseModel, Field, model_validator, field_validator, UUID4
from pydantic_core import PydanticCustomError

from composio import ComposioToolSet, Action, App, action

from versionhq.tool import ComposioAppName, ComposioAuthScheme, composio_app_set, COMPOSIO_STATUS

load_dotenv(override=True)

DEFAULT_REDIRECT_URL = os.environ.get("DEFAULT_REDIRECT_URL", None)
DEFAULT_USER_ID = os.environ.get("DEFAULT_USER_ID", None)


class Composio(BaseModel):
    """
    Class to handle composio tools.
    """

    id: UUID4 = Field(default_factory=uuid.uuid4, frozen=True)
    app_name: str = Field(default=ComposioAppName.HUBSPOT)
    user_id: str = Field(default=DEFAULT_USER_ID)
    auth_scheme: str = Field(default=ComposioAuthScheme.OAUTH2)
    redirect_url: str = Field(default=DEFAULT_REDIRECT_URL, description="redirect url after successful oauth2 connection")
    connect_request_id: str = Field(default=None, description="store the client's composio id to connect with the app")

    @property
    def toolset(self) -> ComposioToolSet:
        return ComposioToolSet(api_key=os.environ.get("COMPOSIO_API_KEY"))


    def __name__(self):
        return self.app_name


    @field_validator("id", mode="before")
    @classmethod
    def _deny_user_set_id(cls, v: Optional[UUID4]) -> None:
        if v:
            raise PydanticCustomError("may_not_set_field", "This field is not to be set by the user.", {})


    # @model_validator("user_id", mode="before")
    # @classmethod
    # def _deny_no_user_id(cls, v: Optional[str]) -> None:
    #     if v is None:
    #         raise PydanticCustomError("user_id_missing", "Need user_id to connect with the tool", {})


    @model_validator(mode="after")
    def validate_app_name(self):
        if self.app_name not in ComposioAppName:
            raise PydanticCustomError("no_app_name", f"The app name {self.app_name} is not valid.", {})

        return self


    @model_validator(mode="after")
    def validate_auth_scheme(self):
        """
        Raise error when the client uses auth scheme unavailable for the app.
        """
        app_set = next(filter(lambda tup: self.app_name in tup, composio_app_set), None)
        if not app_set:
            raise  PydanticCustomError("no_app_set", f"The app set of {self.app_name} is missing.", {})

        else:
            acceptable_auth_scheme = next(filter(lambda item: self.auth_scheme in item, app_set), None)
            if acceptable_auth_scheme is None:
                raise  PydanticCustomError("invalid_auth_scheme", f"The app {self.app_name} must have different auth_scheme.", {})

        return self


    # connect with composio to use the tool
    def connect(self, token: Optional[str] = None, api_key: Optional[str] = None) -> Tuple[str | Any]:
        """
        Connect with Composio, retrieve `connect_request_id`, and validate the connection.
        """

        if not self.user_id:
            raise PydanticCustomError("user_id_missing", "Need user_id to connect with the tool", {})


        connection_request, connected_account = None, None

        if self.auth_scheme == ComposioAuthScheme.API_KEY:
            collected_from_user = {}
            collected_from_user["api_key"] = api_key
            connection_request = self.toolset.initiate_connection(
                connected_account_params = collected_from_user,
                app=self.app_name,
                entity_id=self.user_id,
                auth_scheme=self.auth_scheme,
            )

        if self.auth_scheme == ComposioAuthScheme.BEARER_TOKEN:
            collected_from_user = {}
            collected_from_user["token"] = token
            connection_request = self.toolset.initiate_connection(
                connected_account_params = collected_from_user,
                app=self.app_name,
                entity_id=self.user_id,
                auth_scheme=self.auth_scheme,
            )

        if self.auth_scheme == ComposioAuthScheme.OAUTH2:
            connection_request = self.toolset.initiate_connection(
                app=self.app_name,
                redirect_url = self.redirect_url, # clients will be redirected to this url after successful auth.
                entity_id=self.user_id,
                auth_scheme=self.auth_scheme,
            )

        # connection_request.wait_until_active(self.toolset.client, timeout=60)

        if connection_request.connectionStatus is not COMPOSIO_STATUS.FAILED:
            self.connect_request_id = connection_request.connectedAccountId
            connected_account = self.toolset.get_connected_account(id=self.connect_request_id)

            if connected_account.status is not COMPOSIO_STATUS.FAILED:
                setattr(self.toolset, "entity_id", self.user_id)

        return connected_account, connected_account.status if connected_account else connection_request.connectionStatus




    # @action(toolname=ComposioAppName.HUBSPOT)
    # def deploy(self, param1: str, param2: str, execute_request: Callable) -> str:
    #     """
    #     Define custom actions
    #     my custom action description which will be passed to llm

    #     :param param1: param1 description which will be passed to llm
    #     :param param2: param2 description which will be passed to llm
    #     :return info: return description
    #     """

    #     response = execute_request(
    #         "/my_action_endpoint",
    #         "GET",
    #         {} # body can be added here
    #     )    # execute requests by appending credentials to the request
    #     return str(response) # complete auth dict is available for local use if needed
