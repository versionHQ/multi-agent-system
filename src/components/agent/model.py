import os
import uuid
from abc import ABC
from typing import Any, Dict, List, Optional, TypeVar, Union

from dotenv import load_dotenv
from pydantic import UUID4, BaseModel, Field, InstanceOf, PrivateAttr, model_validator

from components._utils.cache_handler import CacheHandler
from components._utils.logger import Logger, Printer
from components._utils.rpm_controller import RPMController
from components._utils.usage_metrics import UsageMetrics
from components.llm.llm_vars import LLM_VARS
from components.llm.model import LLM
from components.task import OutputFormat
from components.task.model import ResponseField
from components.tool.model import Tool
from components.tool.tool_handler import ToolsHandler

load_dotenv(override=True)
T = TypeVar("T", bound="Agent")

# def _format_answer(agent, answer: str) -> Union[AgentAction, AgentFinish]:
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

    def sum_prompt_tokens(self, tokens: int):
        self.prompt_tokens = self.prompt_tokens + tokens
        self.total_tokens = self.total_tokens + tokens

    def sum_completion_tokens(self, tokens: int):
        self.completion_tokens = self.completion_tokens + tokens
        self.total_tokens = self.total_tokens + tokens

    def sum_cached_prompt_tokens(self, tokens: int):
        self.cached_prompt_tokens = self.cached_prompt_tokens + tokens

    def sum_successful_requests(self, requests: int):
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
class Agent(ABC, BaseModel):
    """
    Base class for third-party agents that LLM runs.
    The agent can execute tasks alone or team. When the agent belongs to team, it needs to prioritize the team.
    * (Temp) Comment out all the optional fields except for Team and LLM settings for convenience.
    """

    __hash__ = object.__hash__
    _logger: Logger = PrivateAttr(default_factory=lambda: Logger(verbose=False))
    _rpm_controller: Optional[RPMController] = PrivateAttr(default=None)
    _request_within_rpm_limit: Any = PrivateAttr(default=None)
    _token_process: TokenProcess = PrivateAttr(default_factory=TokenProcess)
    _times_executed: int = PrivateAttr(default=0)

    id: UUID4 = Field(default_factory=uuid.uuid4, frozen=True)
    # i18n: I18N = Field(default=I18N(), description="Internationalization settings.")
    agent_ops_agent_name: str = None
    agent_ops_agent_id: str = None
    role: str = Field(description="Role of the agent")
    goal: str = Field(description="goal of the agent")
    backstory: str = Field(description="Backstory of the agent")
    # tools: Optional[List[Any]] = Field(default_factory=list, description="Tools at agents' disposal")
    tools_handler: InstanceOf[ToolsHandler] = Field(default=None, description="An instance of the ToolsHandler class")
    # tools_results: Optional[List[Any]] = Field(default=[], description="Results of the tools used by the agent.")
    team: Optional[List[Any]] = Field(default=None, description="Team to which the agent belongs")
    allow_delegation: bool = Field(default=False, description="Enable agent to delegate and ask questions among each other")
    agent_executor: InstanceOf = Field(default=None)
    allow_code_execution: Optional[bool] = Field(default=False, description="Enable code execution for the agent.")

    # llm settings
    llm: Union[str, InstanceOf[LLM], Any] = Field(default=None)
    respect_context_window: bool = Field(default=True, description="Keep messages under the context window size by summarizing content")
    max_retry_limit: int = Field(default=2, description="max. number of retries for an agent to execute a task when an error occurs")
    max_tokens: Optional[int] = Field(default=None, description="max. number of tokens for the agent's execution")
    max_execution_time: Optional[int] = Field(default=None, description="max. execution time for an agent to execute a task")
    max_rpm: Optional[int] = Field(default=None, description="max. number of requests per minute for the agent execution")
    max_iter: Optional[int] = Field(default=25, description="max. number of iterations for an agent to execute a task")
    step_callback: Optional[Any] = Field(default=None, description="Callback to be executed after each step of the agent execution")
    use_system_prompt: Optional[bool] = Field(default=True, description="Use system prompt for the agent")
    function_calling_llm: Optional[Any] = Field(default=None, description="Language model that will run the agent")
    system_template: Optional[str] = Field(default=None, description="System format for the agent.")
    prompt_template: Optional[str] = Field(default=None, description="Prompt format for the agent.")
    response_template: Optional[str] = Field(default=None, description="Response format for the agent.")

    # config, cache, error handling
    config: Optional[Dict[str, Any]] = Field(default=None, exclude=True, description="Configuration for the agent")
    cache: bool = Field(default=True, description="Whether the agent should use a cache for tool usage.")
    cache_handler: InstanceOf[CacheHandler] = Field(default=None, description="An instance of the CacheHandler class.")
    formatting_errors: int = Field(default=0, description="Number of formatting errors.")
    verbose: bool = Field(default=True, description="Verbose mode for the Agent Execution")

    def __repr__(self):
        return f"Agent(role={self.role}, goal={self.goal}, backstory={self.backstory})"

    @model_validator(mode="after")
    def post_init_setup(self):
        """
        Set up the base model and function calling model (if any) using the LLM class.
        The base model is passed from the client app, else use the default model.
        """

        self.agent_ops_agent_name = self.role
        unaccepted_attributes = [
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
            "AWS_REGION_NAME",
        ]

        if isinstance(self.llm, LLM):
            pass

        elif isinstance(self.llm, str):
            self.llm = LLM(model=self.llm)

        elif self.llm is None:
            model_name = os.environ.get(
                "LITELLM_MODEL_NAME", os.environ.get("MODEL", "gpt-4o-mini")
            )
            llm_params = {"model": model_name}
            api_base = os.environ.get(
                "OPENAI_API_BASE", os.environ.get("OPENAI_BASE_URL", None)
            )
            if api_base:
                llm_params["base_url"] = api_base

            set_provider = model_name.split("/")[0] if "/" in model_name else "openai"
            for provider, env_vars in LLM_VARS.items():
                if provider == set_provider:
                    for env_var in env_vars:
                        key_name = env_var.get("key_name")

                        if key_name and key_name not in unaccepted_attributes:
                            env_value = os.environ.get(key_name)
                            if env_value:
                                key_name = (
                                    "api_key" if "API_KEY" in key_name else key_name
                                )
                                key_name = (
                                    "api_base" if "API_BASE" in key_name else key_name
                                )
                                key_name = (
                                    "api_version"
                                    if "API_VERSION" in key_name
                                    else key_name
                                )
                                llm_params[key_name] = env_value
                        elif env_var.get("default", False):
                            for key, value in env_var.items():
                                if key not in ["prompt", "key_name", "default"]:
                                    if key in os.environ:
                                        llm_params[key] = value
            self.llm = LLM(**llm_params)

        else:
            llm_params = {
                "model": getattr(self.llm, "model_name", None)
                or getattr(self.llm, "deployment_name", None)
                or str(self.llm),
                "temperature": getattr(self.llm, "temperature", None),
                "max_tokens": getattr(self.llm, "max_tokens", None),
                "logprobs": getattr(self.llm, "logprobs", None),
                "timeout": getattr(self.llm, "timeout", None),
                "max_retries": getattr(self.llm, "max_retries", None),
                "api_key": getattr(self.llm, "api_key", None),
                "base_url": getattr(self.llm, "base_url", None),
                "organization": getattr(self.llm, "organization", None),
            }
            llm_params = {
                k: v for k, v in llm_params.items() if v is not None
            }  # factor out None values
            self.llm = LLM(**llm_params)

        if self.function_calling_llm:
            if isinstance(self.function_calling_llm, str):
                self.function_calling_llm = LLM(model=self.function_calling_llm)

            elif not isinstance(self.function_calling_llm, LLM):
                self.function_calling_llm = LLM(
                    model=getattr(self.function_calling_llm, "model_name", None)
                    or getattr(self.function_calling_llm, "deployment_name", None)
                    or str(self.function_calling_llm)
                )

        return self

    def invoke(
        self,
        prompts: str,
        output_formats: List[OutputFormat],
        response_fields=List[ResponseField],
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Receive the system prompt in string and create formatted prompts using the system prompt and the agent's backstory.
        Then call the base model.
        """

        messages = []
        messages.append({"role": "user", "content": prompts})  #! REFINEME
        messages.append({"role": "assistant", "content": self.backstory})
        print("Messages sent to the model:", messages)

        callbacks = kwargs.get("callbacks", None)

        response = self.llm.call(
            messages=messages,
            output_formats=output_formats,
            field_list=response_fields,
            callbacks=callbacks,
        )
        print("Agent's answer", response)

        if response is None or response == "":
            Printer.print(
                content="Received None or empty response from LLM call.", color="red"
            )
            raise ValueError("Invalid response from LLM call - None or empty.")

        return {"output": response.output if hasattr(response, "output") else response}

    def execute_task(
        self, task, context: Optional[str] = None, tools: Optional[List[Tool]] = []
    ) -> str:
        """
        Let the agent execute the task and return the output in string.
        When the tool/s are given, the agent must use them.
        The agent must consider the context to excute the task as well when it is given.
        """

        if self.tools_handler:
            self.tools_handler.last_used_tool = {}

        task_prompt = task.prompt()  # return description + exepected output in string

        # if context:
        #     task_prompt = self.i18n.slice("task_with_context").format(task=task_prompt, context=context)

        # tools = tools or self.tools or []
        # self.create_agent_executor(tools=tools, task=task)

        # if self.team and self.team._train:
        #     task_prompt = self._training_handler(task_prompt=task_prompt)
        # else:
        #     task_prompt = self._use_trained_data(task_prompt=task_prompt)

        try:
            result = self.invoke(
                prompts=task_prompt,
                output_formats=task.expected_output_formats,
                response_fields=task.output_json_field_list,
            )["output"]

        except Exception as e:
            self._times_executed += 1
            if self._times_executed > self.max_retry_limit:
                raise e
            result = self.execute_task(task, context, tools)

        if self.max_rpm and self._rpm_controller:
            self._rpm_controller.stop_rpm_counter()

        # for tool_result in self.tools_results:
        #     if tool_result.get("result_as_answer", False):
        #         result = tool_result["result"]

        return result
