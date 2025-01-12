import os
import uuid
from typing import Any, Dict, List, Optional, TypeVar, Callable
from typing_extensions import Self
from dotenv import load_dotenv
from pydantic import UUID4, BaseModel, Field, InstanceOf, PrivateAttr, model_validator, field_validator
from pydantic_core import PydanticCustomError

from versionhq._utils.logger import Logger
from versionhq._utils.rpm_controller import RPMController
from versionhq._utils.usage_metrics import UsageMetrics
from versionhq.llm.llm_variables import LLM_VARS
from versionhq.llm.model import LLM, DEFAULT_CONTEXT_WINDOW_SIZE, DEFAULT_MODEL_NAME
from versionhq.task import TaskOutputFormat
from versionhq.task.model import ResponseField
from versionhq.tool.model import Tool, ToolSet

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
    backstory: Optional[str] = Field(default=None, description="system context passed to the LLM")
    knowledge: Optional[str] = Field(default=None, description="external knowledge fed to the agent")
    skillsets: Optional[List[str]] = Field(default_factory=list)
    tools: Optional[List[Tool | Any]] = Field(default_factory=list)

    # team, task execution rules
    team: Optional[List[Any]] = Field(default=None, description="Team to which the agent belongs")
    allow_delegation: bool = Field(default=False,description="Enable agent to delegate and ask questions among each other")
    allow_code_execution: Optional[bool] = Field(default=False, description="Enable code execution for the agent.")
    max_retry_limit: int = Field(default=2,description="max. number of retries for the task execution when an error occurs. cascaed to the `invoke` function")
    max_iter: Optional[int] = Field(default=25,description="max. number of iterations for an agent to execute a task")
    step_callback: Optional[Callable | Any] = Field(default=None, description="callback to be executed after each step of the agent execution")

    # llm settings cascaded to the LLM model
    llm: str | InstanceOf[LLM] | Dict[str, Any] = Field(default=None)
    function_calling_llm: str | InstanceOf[LLM] | Dict[str, Any] = Field(default=None)
    respect_context_window: bool = Field(default=True,description="Keep messages under the context window size by summarizing content")
    max_tokens: Optional[int] = Field(default=None, description="max. number of tokens for the agent's execution")
    max_execution_time: Optional[int] = Field(default=None, description="max. execution time for an agent to execute a task")
    max_rpm: Optional[int] = Field(default=None, description="max. number of requests per minute for the agent execution")

    # prompt rules
    use_system_prompt: Optional[bool] = Field(default=True, description="Use system prompt for the agent")
    system_template: Optional[str] = Field(default=None, description="System format for the agent.")
    prompt_template: Optional[str] = Field(default=None, description="Prompt format for the agent.")
    response_template: Optional[str] = Field(default=None, description="Response format for the agent.")

    # config, cache, error handling
    formatting_errors: int = Field(default=0, description="Number of formatting errors.")
    agent_ops_agent_name: str = None
    agent_ops_agent_id: str = None


    @field_validator("id", mode="before")
    @classmethod
    def _deny_user_set_id(cls, v: Optional[UUID4]) -> None:
        if v:
            raise PydanticCustomError("may_not_set_field", "This field is not to be set by the user.", {})


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
        Pass the model config params: `llm`, `max_tokens`, `max_execution_time`, `step_callback`,`respect_context_window` to the LLM class.
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


    @model_validator(mode="after")
    def set_up_tools(self) -> Self:
        """
        Similar to the LLM set up, when the agent has tools, we will declare them using the Tool class.
        """
        if not self.tools:
            pass

        else:
            tool_list = []
            def empty_func():
                return "empty function"

            for item in self.tools:
                if isinstance(item, Tool):
                    tool_list.append(item)

                elif isinstance(item, dict):
                    if "function" not in item:
                       setattr(item, "function", empty_func)
                    tool = Tool(**item)
                    tool_list.append(tool)

                elif isinstance(item, str):
                    tool = Tool(name=item, function=empty_func)
                    tool_list.append(tool)

                else:
                    tool_list.append(item) # address custom tool

            self.tools = tool_list

        return self


    @model_validator(mode="after")
    def set_up_backstory(self) -> Self:
        """
        Set up the backstory using a templated BACKSTORY when the backstory is None
        """

        if self.backstory is None:
            from versionhq.agent.TEMPLATES.Backstory import BACKSTORY
            backstory = BACKSTORY.format(
                role=self.role,
                knowledge=self.knowledge if isinstance(self.knowledge, str) else None,
                skillsets=", ".join([item for item in self.skillsets]),
                rag_tool_overview=", ".join([item.name for item in self.tools if hasattr(item, "name")]) if self.tools else "",
                goal=self.goal,
            )
            self.backstory = backstory

        return self


    def _set_llm_params(self, llm: LLM, kwargs: Dict[str, Any] = None) -> LLM:
        """
        After setting up an LLM instance, add params to the instance.
        Prioritize the agent's settings over the model's base setups.
        """

        llm.timeout = self.max_execution_time if llm.timeout is None else llm.timeout
        llm.max_tokens = self.max_tokens if self.max_tokens else llm.max_tokens

        if self.step_callback is not None:
            llm.callbacks = [self.step_callback, ]
            llm._set_callbacks(llm.callbacks)

        if self.respect_context_window == False:
            llm.context_window_size = DEFAULT_CONTEXT_WINDOW_SIZE

        if kwargs:
            for k, v in kwargs.items():
                try:
                    setattr(llm, k, v)
                except:
                    pass
        return llm


    def invoke(self, prompts: str, output_formats: List[str | TaskOutputFormat], response_fields: List[ResponseField]) -> Dict[str, Any]:
        """
        Receive the system prompt in string and create formatted prompts using the system prompt and the agent's backstory.
        Then call the base model.
        When encountering errors, we try the task execution up to `self.max_retry_limit` times.
        """

        task_execution_counter, raw_response = 0, None

        messages = []
        messages.append({"role": "user", "content": prompts})  #! REFINEME
        messages.append({"role": "assistant", "content": self.backstory})
        self._logger.log(level="info", message=f"Messages sent to the model: {messages}", color="blue")

        raw_response = self.llm.call(messages=messages, output_formats=output_formats, field_list=response_fields)
        task_execution_counter += 1
        self._logger.log(level="info", message=f"Agent's first response in {type(raw_response).__name__}: {raw_response}", color="blue")

        if (raw_response is None or raw_response == "") and task_execution_counter < self.max_retry_limit:
            while task_execution_counter <= self.max_retry_limit:
                raw_response = self.llm.call(messages=messages, output_formats=output_formats, field_list=response_fields)
                task_execution_counter += 1
                self._logger.log(level="info", message=f"Agent's next response in {type(raw_response).__name__}: {raw_response}", color="blue")

        elif raw_response is None or raw_response == "":
            self._logger.log(level="error", message="Received None or empty response from the model", color="red")
            raise ValueError("Invalid response from LLM call - None or empty.")

        return raw_response


    def execute_task(self, task, context: Optional[str] = None) -> str:
        """
        Execute the task and return the response in string.
        The agent utilizes the tools in task or their own tools if the task.can_use_agent_tools is True.
        The agent must consider the context to excute the task as well when it is given.
        """

        task_prompt = task.prompt()
        if context is not task.prompt_context:  # as `task.prompt()` includes adding `task.prompt_context` to the prompt.
            task_prompt += context

        tool_results = []
        if task.tools:
            for item in task.tools:
                if isinstance(item, ToolSet):
                    tool_result = item.tool.run(**item.kwargs)
                    tool_results.append(tool_result)
                elif isinstance(item, Tool):
                    tool_result = item.run()
                    tool_results.append(tool_result)
                else:
                    try:
                        item.run()
                    except:
                        pass

        if task.can_use_agent_tools is True and self.tools:
            for tool in self.tools:
                tool_result = tool.run()
                tool_results.append(tool_result)

        if task.take_tool_res_as_final:
            return tool_results

        # if self.team and self.team._train:
        #     task_prompt = self._training_handler(task_prompt=task_prompt)
        # else:
        #     task_prompt = self._use_trained_data(task_prompt=task_prompt)

        try:
            raw_response = self.invoke(
                prompts=task_prompt,
                output_formats=task.expected_output_formats,
                response_fields=task.output_field_list,
            )

        except Exception as e:
            self._times_executed += 1
            if self._times_executed > self.max_retry_limit:
                raise e
            raw_response = self.execute_task(task, context)

        if self.max_rpm and self._rpm_controller:
            self._rpm_controller.stop_rpm_counter()

        return raw_response


    def __repr__(self):
        return f"Agent(role={self.role}, goal={self.goal}, backstory={self.backstory})"
