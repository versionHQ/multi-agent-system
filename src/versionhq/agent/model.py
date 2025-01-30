import os
import uuid
import datetime
from typing import Any, Dict, List, Optional, TypeVar, Callable, Type
from typing_extensions import Self
from dotenv import load_dotenv
import litellm

from pydantic import UUID4, BaseModel, Field, InstanceOf, PrivateAttr, model_validator, field_validator, ConfigDict
from pydantic_core import PydanticCustomError

from versionhq.llm.model import LLM, DEFAULT_CONTEXT_WINDOW_SIZE, DEFAULT_MODEL_NAME
from versionhq.tool.model import Tool, ToolSet
from versionhq.knowledge.model import BaseKnowledgeSource, Knowledge
from versionhq.memory.contextual_memory import ContextualMemory
from versionhq.memory.model import ShortTermMemory, LongTermMemory, UserMemory
from versionhq._utils.logger import Logger
from versionhq.agent.rpm_controller import RPMController
from versionhq._utils.usage_metrics import UsageMetrics
from versionhq._utils.process_config import process_config


load_dotenv(override=True)
T = TypeVar("T", bound="Agent")


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
    skillsets: Optional[List[str]] = Field(default_factory=list)
    tools: Optional[List[InstanceOf[Tool | ToolSet] | Type[Tool] | Any]] = Field(default_factory=list)

    # knowledge
    knowledge_sources: Optional[List[BaseKnowledgeSource]] = Field(default=None)
    _knowledge: Optional[Knowledge] = PrivateAttr(default=None)

    # memory
    use_memory: bool = Field(default=False, description="whether to store/use memory when executing the task")
    memory_config: Optional[Dict[str, Any]] = Field(default=None, description="configuration for the memory. need to store user_id for UserMemory")
    short_term_memory: Optional[InstanceOf[ShortTermMemory]] = Field(default=None)
    long_term_memory: Optional[InstanceOf[LongTermMemory]] = Field(default=None)
    user_memory: Optional[InstanceOf[UserMemory]] = Field(default=None)
    embedder_config: Optional[Dict[str, Any]] = Field(default=None, description="embedder configuration for the agent's knowledge")

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
    max_rpm: Optional[int] = Field(default=None, description="max. number of requests per minute")
    llm_config: Optional[Dict[str, Any]] = Field(default=None, description="other llm config cascaded to the model")

    # cache, error, ops handling
    formatting_errors: int = Field(default=0, description="number of formatting errors.")
    agent_ops_agent_name: str = None
    agent_ops_agent_id: str = None


    @field_validator("id", mode="before")
    @classmethod
    def _deny_user_set_id(cls, v: Optional[UUID4]) -> None:
        if v:
            raise PydanticCustomError("may_not_set_field", "This field is not to be set by the user.", {})


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

        if self.callbacks:
            llm.callbacks = self.callbacks
            llm._set_callbacks(llm.callbacks)

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
            skills = ", ".join([item for item in self.skillsets]) if self.skillsets else ""
            tools = ", ".join([item.name for item in self.tools if hasattr(item, "name")]) if self.tools else ""
            role = self.role.lower()
            goal = self.goal.lower()

            if self.tools or self.skillsets:
                backstory = BACKSTORY_FULL.format(role=role, goal=goal, skills=skills, tools=tools)
            else:
                backstory = BACKSTORY_SHORT.format(role=role, goal=goal)

            self.backstory = backstory

        return self


    @model_validator(mode="after")
    def set_up_rpm(self) -> Self:
        """
        Set up RPM controller.
        """
        if self.max_rpm:
            self._rpm_controller = RPMController(max_rpm=self.max_rpm, _current_rpm=0)

        return self


    @model_validator(mode="after")
    def set_up_knowledge(self) -> Self:
        if self.knowledge_sources:
            knowledge_agent_name = f"{self.role.replace(' ', '_')}"

            if isinstance(self.knowledge_sources, list) and all(isinstance(k, BaseKnowledgeSource) for k in self.knowledge_sources):
                self._knowledge = Knowledge(
                    sources=self.knowledge_sources,
                    embedder_config=self.embedder_config,
                    collection_name=knowledge_agent_name,
                )

        return self


    @model_validator(mode="after")
    def set_up_memory(self) -> Self:
        """
        Set up memories: stm, um
        """

        if self.use_memory == True:
            self.long_term_memory = self.long_term_memory if self.long_term_memory else LongTermMemory()
            self.short_term_memory = self.short_term_memory if self.short_term_memory else ShortTermMemory(agent=self, embedder_config=self.embedder_config)

            if hasattr(self, "memory_config") and self.memory_config is not None:
                user_id = self.memory_config.get("user_id", None)
                if user_id:
                    self.user_memory = self.user_memory if self.user_memory else UserMemory(agent=self, user_id=user_id)
            else:
                self.user_memory = None

        return self


    def _train(self) -> Self:
        """
        Fine-tuned the base model using OpenAI train framework.
        """
        if not isinstance(self.llm, LLM):
            pass


    def invoke(
        self,
        prompts: str,
        response_format: Optional[Dict[str, Any]] = None,
        tools: Optional[List[InstanceOf[Tool]| InstanceOf[ToolSet] | Type[Tool]]] = None,
        tool_res_as_final: bool = False,
        task: Any = None
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
        messages.append({ "role": "user", "content": prompts })
        if self.use_developer_prompt:
            messages.append({ "role": "system", "content": self.backstory })

        try:
            if self._rpm_controller and self.max_rpm:
                self._rpm_controller.check_or_wait()

            self._logger.log(level="info", message=f"Messages sent to the model: {messages}", color="blue")

            if tool_res_as_final:
                func_llm = self.function_calling_llm if self.function_calling_llm and self.function_calling_llm._supports_function_calling() else LLM(model=DEFAULT_MODEL_NAME)
                raw_response = func_llm.call(messages=messages, tools=tools, tool_res_as_final=True)
                task.tokens = func_llm._tokens
            else:
                raw_response = self.llm.call(messages=messages, response_format=response_format, tools=tools)
                task.tokens = self.llm._tokens

            task_execution_counter += 1
            self._logger.log(level="info", message=f"Agent response: {raw_response}", color="blue")
            return raw_response

        except Exception as e:
            self._logger.log(level="error", message=f"An error occured. The agent will retry: {str(e)}", color="red")

            while not raw_response and task_execution_counter <= self.max_retry_limit:
                while (not raw_response or raw_response == "" or raw_response is None) and iterations < self.maxit:
                    if self.max_rpm and self._rpm_controller:
                        self._rpm_controller.check_or_wait()

                    raw_response = self.llm.call(messages=messages, response_format=response_format, tools=tools)
                    task.tokens = self.llm._tokens
                    iterations += 1

                task_execution_counter += 1
                self._logger.log(level="info", message=f"Agent #{task_execution_counter} response: {raw_response}", color="blue")
                return raw_response

            if not raw_response:
                self._logger.log(level="error", message="Received None or empty response from the model", color="red")
                raise ValueError("Invalid response from LLM call - None or empty.")




    def execute_task(self, task, context: Optional[str] = None, task_tools: Optional[List[Tool | ToolSet]] = list()) -> str:
        """
        Execute the task and return the response in string.
        The agent utilizes the tools in task or their own tools if the task.can_use_agent_tools is True.
        The agent must consider the context to excute the task as well when it is given.
        """
        from versionhq.task.model import Task
        from versionhq.knowledge._utils import extract_knowledge_context

        task: InstanceOf[Task] = task
        tools: Optional[List[InstanceOf[Tool]| InstanceOf[ToolSet] | Type[Tool]]] = task_tools + self.tools if task.can_use_agent_tools else task_tools

        if self.max_rpm and self._rpm_controller:
            self._rpm_controller._reset_request_count()

        task_prompt = task.prompt(model_provider=self.llm.provider)
        if context is not task.prompt_context:
            task_prompt += context

        if self._knowledge:
            agent_knowledge = self._knowledge.query(query=[task_prompt,])
            if agent_knowledge:
                agent_knowledge_context = extract_knowledge_context(knowledge_snippets=agent_knowledge)
                if agent_knowledge_context:
                    task_prompt += agent_knowledge_context


        if self.use_memory == True:
            contextual_memory = ContextualMemory(
                memory_config=self.memory_config, stm=self.short_term_memory, ltm=self.long_term_memory, um=self.user_memory
            )
            memory = contextual_memory.build_context_for_task(task=task, context=context)
            if memory.strip() != "":
                task_prompt += memory.strip()


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
                task=task
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
