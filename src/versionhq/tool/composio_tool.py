import os
import uuid
from abc import ABC
from dotenv import load_dotenv
from typing import Any, Callable, Type, get_args, get_origin, Optional, Tuple, Dict
from typing_extensions import Self

from pydantic import BaseModel, Field, model_validator, field_validator, UUID4, PrivateAttr
from pydantic_core import PydanticCustomError

from composio import ComposioToolSet
from composio_langchain import action

from versionhq.tool import ComposioAppName, ComposioAuthScheme, composio_app_set, ComposioStatus, ComposioAction
from versionhq._utils.logger import Logger
from versionhq._utils.cache_handler import CacheHandler

load_dotenv(override=True)

DEFAULT_REDIRECT_URL = os.environ.get("DEFAULT_REDIRECT_URL", None)
DEFAULT_USER_ID = os.environ.get("DEFAULT_USER_ID", None)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", None)


class ComposioHandler(ABC, BaseModel):
    """
    A class to handle connecting account with Composio and executing actions using Composio ecosystem.
    `connected_account_id` is set up per `app_name` to call the actions on the given app. i.e., salesforce
    """

    _logger: Logger = PrivateAttr(default_factory=lambda: Logger(verbose=True))
    _cache: CacheHandler = PrivateAttr(default_factory=lambda: CacheHandler())

    id: UUID4 = Field(default_factory=uuid.uuid4, frozen=True)
    app_name: str = Field(default=ComposioAppName.HUBSPOT, max_length=128, description="app name defined by composio")
    user_id: str = Field(default=DEFAULT_USER_ID, description="composio entity id")
    auth_scheme: str = Field(default=ComposioAuthScheme.OAUTH2)
    redirect_url: str = Field(default=DEFAULT_REDIRECT_URL, description="redirect url after successful oauth2 connection")
    connected_account_id: str = Field(
        default=None,
        description="store the client id generated by composio after auth validation. use the id to connect with a given app and execute composio actions"
    )
    tools: Any = Field(default=None, descritpion="retrieved composio tools")

    @field_validator("id", mode="before")
    @classmethod
    def _deny_user_set_id(cls, v: Optional[UUID4]) -> None:
        if v:
            raise PydanticCustomError("may_not_set_field", "This field is not to be set by the user.", {})


    @model_validator(mode="after")
    def validate_app_name(self):
        if self.app_name not in ComposioAppName:
            raise PydanticCustomError("no_app_name", f"The given app name {self.app_name} is not valid.", {})

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


    def _setup_langchain_toolset(self, metadata: Dict[str, Any] = dict()):
        """
        Composio toolset on LangChain for action execution using LLM.
        """
        from composio_langchain import ComposioToolSet
        return ComposioToolSet(api_key=os.environ.get("COMPOSIO_API_KEY"), metadata={**metadata})


    def _connect(
            self, token: Optional[str] = None, api_key: Optional[str] = None, connected_account_id: str = None
        ) -> Tuple[Self | str | Any]:
        """
        Send connection request to Composio, retrieve `connected_account_id`, and proceed with OAuth process of the given app to activate the connection.
        """

        connection_request, connected_account = None, None
        connected_account_id = connected_account_id or self.connected_account_id
        if connected_account_id:
            connected_account = self.toolset.get_connected_account(id=connected_account_id)

            if connected_account and connected_account.status == ComposioStatus.ACTIVE.value:
                return self, ComposioStatus.ACTIVE.value

        if not self.user_id:
            raise PydanticCustomError("entity_id_missing", "Need entity_id to connect with the tool", {})

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

        if connection_request.connectionStatus == ComposioStatus.FAILED.value:
            self._logger.log(level="error", message="Connection to composio failed.", color="red")
            raise PydanticCustomError("connection_failed", "Connection to composio has failed", {})


        connected_account = self.toolset.get_connected_account(id=connection_request.connectedAccountId)
        # Note: connected_account.id === connection_request.connectedAccountId === self.connected_account_id

        if connected_account.status == ComposioStatus.ACTIVE.value:
            setattr(self.toolset, "entity_id", self.user_id)
            self.connected_account_id = connection_request.connectedAccountId

        elif connected_account.status == ComposioStatus.INITIATED.value:
            setattr(self.toolset, "entity_id", self.user_id)
            self.connected_account_id = connection_request.connectedAccountId

            if connection_request.redirectUrl:
                import webbrowser
                webbrowser.open(connection_request.redirectUrl)

        else:
            self._logger.log(level="error", message="The account is invalid.", color="red")
            raise PydanticCustomError("connection_failed", "Connection to composio has failed", {})

        return self, connected_account.status if connected_account else connection_request.connectionStatus


    def execute_composio_action_with_langchain(self, action_name: str | ComposioAction, task_in_natural_language: str) -> Tuple[Self, str]:
        """
        Execute Composio's Action using Langchain's agent ecosystem.
        """
        from langchain import hub
        from langchain_openai import ChatOpenAI
        from langchain.agents import create_openai_functions_agent, AgentExecutor
        from composio_langchain import Action

        action_name = action_name.value if isinstance(action_name, ComposioAction) else action_name
        action = Action(action_name)
        metadata = { action:  { "OPENAI_API_KEY": OPENAI_API_KEY } }
        toolset = self._setup_langchain_toolset(metadata=metadata)
        tools = toolset.get_tools(actions=[action_name,], entity_id=self.user_id)
        if not tools:
            self._logger.log(level="error", message=f"Tools related to {action_name} are not found on Langchain", color="red")
            raise PydanticCustomError("tool_not_found", "Tools not found on Langchain", {})

        self.tools = tools
        llm = ChatOpenAI()
        prompt = hub.pull("hwchase17/openai-functions-agent")
        agent = create_openai_functions_agent(llm, tools, prompt)
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
        result = agent_executor.invoke(dict(input=task_in_natural_language))
        return self, result["output"]


    @property
    def toolset(self) -> ComposioToolSet:
        return ComposioToolSet(api_key=os.environ.get("COMPOSIO_API_KEY"))


    def __name__(self):
        return self.app_name
