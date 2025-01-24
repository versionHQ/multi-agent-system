import os
import uuid
from typing import Any, Dict, List, Optional, TypeVar, Callable, Type
from typing_extensions import Self
from dotenv import load_dotenv
import litellm

from pydantic import UUID4, BaseModel, Field, InstanceOf, PrivateAttr, model_validator, field_validator, ConfigDict
from pydantic_core import PydanticCustomError

from versionhq.llm.model import LLM, DEFAULT_CONTEXT_WINDOW_SIZE, DEFAULT_MODEL_NAME
from versionhq.tool.model import Tool, ToolSet
from versionhq._utils.logger import Logger
from versionhq._utils.rpm_controller import RPMController
from versionhq._utils.usage_metrics import UsageMetrics
from versionhq._utils.process_config import process_config


load_dotenv(override=True)
T = TypeVar("T", bound="Agent")


# def _format_answer(agent, answer: str) -> AgentAction | AgentFinish:
#     return AgentParser(agent=agent).parse(answer)

# def mock_agent_ops_provider():
#     def track_agent(*args, **kwargs):
#         def noop(f):
#             return f
#         return noop
#     return track_agent

# track_agent = mock_agent_ops_provider()

# agentops = None
# if os.environ.get("AGENTOPS_API_KEY"):
#     try:
#         from agentops import track_agent
#     except ImportError:
#         track_agent = mock_agent_ops_provider()
# else:
#    track_agent = mock_agent_ops_provider()


class TokenProcess:
    total_tokens: int = 0
    prompt_tokens: int = 0
    cached_prompt_tokens: int = 0
    completion_tokens: int = 0
    successful_requests: int = 0

    def sum_prompt_tokens(self, tokens: int) -> None:
        self.prompt_tokens = self.prompt_tokens + tokens
        self.total_tokens = self.total_tokens + tokens

    def sum_completion_tokens(self, tokens: int) -> None:
        self.completion_tokens = self.completion_tokens + tokens
        self.total_tokens = self.total_tokens + tokens

    def sum_cached_prompt_tokens(self, tokens: int) -> None:
        self.cached_prompt_tokens = self.cached_prompt_tokens + tokens

    def sum_successful_requests(self, requests: int) -> None:
        self.successful_requests = self.successful_requests + requests

    def get_summary(self) -> UsageMetrics:
        return UsageMetrics(
            total_tokens=self.total_tokens,
            prompt_tokens=self.prompt_tokens,
            cached_prompt_tokens=self.cached_prompt_tokens,
            completion_tokens=self.completion_tokens,
            successful_requests=self.successful_requests,
        )


# @track_agent()
class Agent(BaseModel):
    """
    Agent class that run on LLM.
    Agents execute tasks alone or in the team, using RAG tools and knowledge base if any.
    Agents will prioritize team tasks when they belong to the team.
    * (Temp) Comment out all the optional fields except for Team and LLM settings for convenience.
    """

    __hash__ = object.__hash__
    _logger: Logger = PrivateAttr(default_factory=lambda: Logger(verbose=True))
    _rpm_controller: Optional[RPMController] = PrivateAttr(default=None)
    _request_within_rpm_limit: Any = PrivateAttr(default=None)
    _token_process: TokenProcess = PrivateAttr(default_factory=TokenProcess)
    _times_executed: int = PrivateAttr(default=0)
    config: Optional[Dict[str, Any]] = Field(default=None, exclude=True, description="values to add to the Agent class")

    id: UUID4 = Field(default_factory=uuid.uuid4, frozen=True)
    role: str = Field(description="role of the agent - used in summary and logs")
    goal: str = Field(description="concise goal of the agent (details are set in the Task instance)")
    backstory: Optional[str] = Field(default=None, description="developer prompt to the llm")
    knowledge: Optional[str] = Field(default=None, description="external knowledge fed to the agent")
    skillsets: Optional[List[str]] = Field(default_factory=list)
    tools: Optional[List[Tool | ToolSet | Type[Tool]]] = Field(default_factory=list)

    # prompting
    use_developer_prompt: Optional[bool] = Field(default=True, description="Use developer prompt when calling the llm")
    developer_propmt_template: Optional[str] = Field(default=None, description="ddeveloper prompt template")
    user_prompt_template: Optional[str] = Field(default=None, description="user prompt template")

    # task execution rules
    team: Optional[List[Any]] = Field(default=None, description="Team to which the agent belongs")
    allow_delegation: bool = Field(default=False,description="if the agent can delegate the task to another agent or ask some help")
    max_retry_limit: int = Field(default=2 ,description="max. number of retry for the task execution when an error occurs")
    maxit: Optional[int] = Field(default=25,description="max. number of total optimization loops conducted when an error occurs")
    callbacks: Optional[List[Callable]] = Field(default_factory=list, description="callback functions to execute after any task execution")

    # llm settings cascaded to the LLM model
    llm: str | InstanceOf[LLM] | Dict[str, Any] = Field(default=None)
    function_calling_llm: str | InstanceOf[LLM] | Dict[str, Any] = Field(default=None)
    respect_context_window: bool = Field(default=True,description="Keep messages under the context window size by summarizing content")
    max_tokens: Optional[int] = Field(default=None, description="max. number of tokens for the agent's execution")
    max_execution_time: Optional[int] = Field(default=None, description="max. execution time for an agent to execute a task")
    max_rpm: Optional[int] = Field(default=None, description="max. number of requests per minute for the agent execution")
    llm_config: Optional[Dict[str, Any]] = Field(default=None, description="other llm config cascaded to the model")

    # config, cache, error handling
    formatting_errors: int = Field(default=0, description="number of formatting errors.")
    agent_ops_agent_name: str = None
    agent_ops_agent_id: str = None


    @field_validator("id", mode="before")
    @classmethod
    def _deny_user_set_id(cls, v: Optional[UUID4]) -> None:
        if v:
            raise PydanticCustomError("may_not_set_field", "This field is not to be set by the user.", {})


    # @field_validator(mode="before")
    # def set_up_from_config(cls) -> None:
    #     if cls.config is not None:
    #         try:
    #             for k, v in cls.config.items():
    #                 setattr(cls, k, v)
    #         except:
    #             pass

    @model_validator(mode="before")
    @classmethod
    def process_model_config(cls, values: Dict[str, Any]) -> None:
        return process_config(values_to_update=values, model_class=cls)


    @model_validator(mode="after")
    def validate_required_fields(self) -> Self:
        required_fields = ["role", "goal"]
        for field in required_fields:
            if getattr(self, field) is None:
                raise ValueError(f"{field} must be provided either directly or through config")
        return self


    @model_validator(mode="after")
    def set_up_llm(self) -> Self:
        """
        Set up the base model and function calling model (if any) using the LLM class.
        Pass the model config params: `llm`, `max_tokens`, `max_execution_time`, `callbacks`,`respect_context_window` to the LLM class.
        The base model is selected on the client app, else use the default model.
        """

        self.agent_ops_agent_name = self.role
        # unaccepted_attributes = ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_REGION_NAME"]

        if isinstance(self.llm, LLM):
            llm = self._set_llm_params(self.llm)
            self.llm = llm

        elif isinstance(self.llm, str) or self.llm is None:
            model_name = self.llm if self.llm is not None else DEFAULT_MODEL_NAME
            llm = LLM(model=model_name)
            updated_llm = self._set_llm_params(llm)
            self.llm = updated_llm

        else:
            if isinstance(self.llm, dict):
                model_name = self.llm.pop("model_name", self.llm.pop("deployment_name", str(self.llm)))
                llm = LLM(model=model_name if model_name is not None else DEFAULT_MODEL_NAME)
                updated_llm = self._set_llm_params(llm, { k: v for k, v in self.llm.items() if v is not None })
                self.llm = updated_llm

            else:
                model_name = (getattr(self.llm, "model_name") or getattr(self.llm, "deployment_name") or str(self.llm))
                llm = LLM(model=model_name)
                llm_params = {
                    "max_tokens": (getattr(self.llm, "max_tokens") or self.max_tokens or 3000),
                    "timeout": getattr(self.llm, "timeout", self.max_execution_time),
                    "callbacks": getattr(self.llm, "callbacks", None),
                    "temperature": getattr(self.llm, "temperature", None),
                    "logprobs": getattr(self.llm, "logprobs", None),
                    "api_key": getattr(self.llm, "api_key", os.environ.get("LITELLM_API_KEY", None)),
                    "base_url": getattr(self.llm, "base_url", None),
                }
                updated_llm = self._set_llm_params(llm, llm_params)
                self.llm = updated_llm


        """
        Set up funcion_calling LLM as well.
        Check if the model supports function calling, setup LLM instance accordingly, using the same params with the LLM.
        """
        if self.function_calling_llm:
            if isinstance(self.function_calling_llm, LLM):
                if self.function_calling_llm._supports_function_calling() == False:
                    self.function_calling_llm = LLM(model=DEFAULT_MODEL_NAME)

                updated_llm = self._set_llm_params(self.function_calling_llm)
                self.function_calling_llm = updated_llm

            elif isinstance(self.function_calling_llm, str):
                llm = LLM(model=self.function_calling_llm)

                if llm._supports_function_calling() == False:
                    llm = LLM(model=DEFAULT_MODEL_NAME)

                updated_llm = self._set_llm_params(llm)
                self.function_calling_llm = updated_llm

            else:
                if isinstance(self.function_calling_llm, dict):
                    model_name = self.function_calling_llm.pop("model_name", self.function_calling_llm.pop("deployment_name", str(self.function_calling_llm)))
                    llm = LLM(model=model_name)
                    updated_llm = self._set_llm_params(llm, { k: v for k, v in self.function_calling_llm.items() if v is not None })
                    self.function_calling_llm = updated_llm

                else:
                    model_name = (getattr(self.function_calling_llm, "model_name") or getattr(self.function_calling_llm, "deployment_name") or str(self.function_calling_llm))
                    llm = LLM(model=model_name)
                    llm_params = {
                        "max_tokens": (getattr(self.function_calling_llm, "max_tokens") or self.max_tokens or 3000),
                        "timeout": getattr(self.function_calling_llm, "timeout", self.max_execution_time),
                        "callbacks": getattr(self.function_calling_llm, "callbacks", None),
                        "temperature": getattr(self.function_calling_llm, "temperature", None),
                        "logprobs": getattr(self.function_calling_llm, "logprobs", None),
                        "api_key": getattr(self.function_calling_llm, "api_key", os.environ.get("LITELLM_API_KEY", None)),
                        "base_url": getattr(self.function_calling_llm, "base_url", None),
                    }
                    updated_llm = self._set_llm_params(llm, llm_params)
                    self.function_calling_llm = updated_llm

        return self


    def _set_llm_params(self, llm: LLM, config: Dict[str, Any] = None) -> LLM:
        """
        After setting up an LLM instance, add params to the instance.
        Prioritize the agent's settings over the model's base setups.
        """

        llm.timeout = self.max_execution_time if llm.timeout is None else llm.timeout
        llm.max_tokens = self.max_tokens if self.max_tokens else llm.max_tokens

        # if self.callbacks:
        #     llm.callbacks = self.callbacks
        #     llm._set_callbacks(llm.callbacks)

        if self.respect_context_window == False:
            llm.context_window_size = DEFAULT_CONTEXT_WINDOW_SIZE

        config = self.config.update(config) if self.config else config
        if config:
            valid_params = litellm.get_supported_openai_params(model=llm.model)
            for k, v in config.items():
                try:
                    if k in valid_params and v is not None:
                        setattr(llm, k, v)
                except:
                    pass
        return llm


    @model_validator(mode="after")
    def set_up_tools(self) -> Self:
        """
        Similar to the LLM set up, when the agent has tools, we will declare them using the Tool class.
        """
        if not self.tools:
            pass

        else:
            tool_list = []

            for item in self.tools:
                if isinstance(item, Tool):
                    tool_list.append(item)

                elif isinstance(item, dict) and "func" in item:
                    tool = Tool(**item)
                    tool_list.append(tool)

                elif type(item) is Tool and hasattr(item, "func"):
                    tool_list.append(item)

                else:
                    self._logger.log(level="error", message=f"Tool {str(item)} is missing a function.", color="red")
                    raise PydanticCustomError("invalid_tool", f"The tool {str(item)} is missing a function.", {})

            self.tools = tool_list

        return self


    @model_validator(mode="after")
    def set_up_backstory(self) -> Self:
        """
        Set up the backstory using a templated BACKSTORY when the backstory is None
        """

        if self.backstory is None:
            from versionhq.agent.TEMPLATES.Backstory import BACKSTORY_FULL, BACKSTORY_SHORT
            backstory = ""

            if self.tools or self.knowledge or self.skillsets:
                backstory = BACKSTORY_FULL.format(
                    role=self.role,
                    goal=self.goal,
                    knowledge=self.knowledge if isinstance(self.knowledge, str) else None,
                    skillsets=", ".join([item for item in self.skillsets]),
                    rag_tool_overview=", ".join([item.name for item in self.tools if hasattr(item, "name")]) if self.tools else "",
                )
            else:
                backstory = BACKSTORY_SHORT.format(role=self.role, goal=self.goal)

            self.backstory = backstory

        return self


    def invoke(
        self,
        prompts: str,
        response_format: Optional[Dict[str, Any]] = None,
        tools: Optional[List[Tool | ToolSet | Type[Tool]]] = None,
        tool_res_as_final: bool = False
        ) -> Dict[str, Any]:
        """
        Create formatted prompts using the developer prompt and the agent's backstory, then call the base model.
        - Execute the task up to `self.max_retry_limit` times in case of receiving an error or empty response.
        - Pass the task_tools to the model to let them execute.
        """

        task_execution_counter = 0
        iterations = 0
        raw_response = None
        messages = []

        messages.append({"role": "user", "content": prompts})
        if self.use_developer_prompt:
            messages.append({"role": "system", "content": self.backstory})
        self._logger.log(level="info", message=f"Messages sent to the model: {messages}", color="blue")

        try:
            if tool_res_as_final is True:
                func_llm = self.function_calling_llm if self.function_calling_llm and self.function_calling_llm._supports_function_calling() else LLM(model=DEFAULT_MODEL_NAME)
                raw_response = func_llm.call(messages=messages, tools=tools, tool_res_as_final=True)
            else:
                raw_response = self.llm.call(messages=messages, response_format=response_format, tools=tools)

            task_execution_counter += 1
            self._logger.log(level="info", message=f"Agent response: {raw_response}", color="blue")

            if raw_response and self.callbacks:
                for item in self.callbacks:
                    raw_response = item(raw_response)

        except Exception as e:
            self._logger.log(level="error", message=f"An error occured. The agent will retry: {str(e)}", color="red")

            while not raw_response and task_execution_counter < self.max_retry_limit:
                while not raw_response and iterations < self.maxit:
                    raw_response = self.llm.call(messages=messages, response_format=response_format, tools=tools)
                    iterations += 1

                task_execution_counter += 1
                self._logger.log(level="info", message=f"Agent #{task_execution_counter} response: {raw_response}", color="blue")

                if raw_response and self.callbacks:
                    for item in self.callbacks:
                        raw_response = item(raw_response)

            if not raw_response:
                self._logger.log(level="error", message="Received None or empty response from the model", color="red")
                raise ValueError("Invalid response from LLM call - None or empty.")

        return raw_response


    def execute_task(self, task, context: Optional[str] = None, task_tools: Optional[List[Tool | ToolSet]] = list()) -> str:
        """
        Execute the task and return the response in string.
        The agent utilizes the tools in task or their own tools if the task.can_use_agent_tools is True.
        The agent must consider the context to excute the task as well when it is given.
        """
        from versionhq.task.model import Task

        task: InstanceOf[Task] = task
        tools: Optional[List[Tool | ToolSet | Type[Tool]]] = task_tools + self.tools if task.can_use_agent_tools else task_tools

        task_prompt = task.prompt(model_provider=self.llm.provider)
        if context is not task.prompt_context:
            task_prompt += context

        # if self.team and self.team._train:
        #     task_prompt = self._training_handler(task_prompt=task_prompt)
        # else:
        #     task_prompt = self._use_trained_data(task_prompt=task_prompt)

        try:
            self._times_executed += 1
            raw_response = self.invoke(
                prompts=task_prompt,
                response_format=task._structure_response_format(model_provider=self.llm.provider),
                tools=tools,
                tool_res_as_final=task.tool_res_as_final,
            )

        except Exception as e:
            self._times_executed += 1
            self._logger.log(level="error", message=f"The agent failed to execute the task. Error: {str(e)}", color="red")
            raw_response = self.execute_task(task, context, task_tools)

            if self._times_executed > self.max_retry_limit:
                self._logger.log(level="error", message=f"Max retry limit has exceeded.", color="red")
                raise e

        if self.max_rpm and self._rpm_controller:
            self._rpm_controller.stop_rpm_counter()

        return raw_response


    def __repr__(self):
        return f"Agent(role={self.role}, goal={self.goal}, backstory={self.backstory})"
