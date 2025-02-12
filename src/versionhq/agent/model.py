import os
import uuid
from typing import Any, Dict, List, Optional, TypeVar, Callable, Type
from typing_extensions import Self
from dotenv import load_dotenv
import litellm

from pydantic import UUID4, BaseModel, Field, InstanceOf, PrivateAttr, model_validator, field_validator
from pydantic_core import PydanticCustomError

from versionhq.llm.model import LLM, DEFAULT_CONTEXT_WINDOW_SIZE, DEFAULT_MODEL_NAME, PROVIDERS
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
    A class to store agent information.
    Agents must have `role`, `goal`, and `llm` = DEFAULT_MODEL_NAME as default.
    Then run validation on `backstory`, `llm`, `tools`, `rpm` (request per min), `knowledge`, and `memory`.
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
    knowledge_sources: Optional[List[BaseKnowledgeSource | Any]] = Field(default=None)
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
    network: Optional[List[Any]] = Field(default=None, description="store a list of agent networks that the agent belong as a member")
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
    llm_config: Optional[Dict[str, Any]] = Field(default=None, description="other llm config cascaded to the LLM model")

    # cache, error, ops handling
    formatting_errors: int = Field(default=0, description="number of formatting errors.")


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
        Set up `llm` and `function_calling_llm` as valid LLM objects using the given kwargs.
        """
        self.llm = self._convert_to_llm_object(llm=self.llm)

        function_calling_llm = self.function_calling_llm if self.function_calling_llm else self.llm if self.llm else None
        function_calling_llm = self._convert_to_llm_object(llm=function_calling_llm)
        if function_calling_llm._supports_function_calling():
            self.function_calling_llm = function_calling_llm
        elif self.llm._supports_function_calling():
            self.function_calling_llm = self.llm
        else:
            self.function_calling_llm = self._convert_to_llm_object(llm=LLM(model=DEFAULT_MODEL_NAME))
        return self


    def _convert_to_llm_object(self, llm: Any = None) -> LLM:
        """
        Convert the given value to LLM object.
        When `llm` is dict or self.llm_config is not None, add these values to the LLM object after validating them.
        """
        llm = llm if llm else self.llm if self.llm else DEFAULT_MODEL_NAME

        if not llm:
            pass

        match llm:
            case LLM():
                return self._set_llm_params(llm=llm, config=self.llm_config)

            case str():
                llm_obj = LLM(model=llm)
                return self._set_llm_params(llm=llm_obj, config=self.llm_config)

            case dict():
                model_name = llm.pop("model_name", llm.pop("deployment_name", str(llm)))
                llm_obj = LLM(model=model_name if model_name else DEFAULT_MODEL_NAME)
                config = llm.update(self.llm_config) if self.llm_config else llm
                return self._set_llm_params(llm_obj, config=config)

            case _:
                model_name = (getattr(self.llm, "model_name") or getattr(self.llm, "deployment_name") or str(self.llm))
                llm_obj = LLM(model=model_name if model_name else DEFAULT_MODEL_NAME)
                llm_params = {
                    "max_tokens": (getattr(llm, "max_tokens") or self.max_tokens or 3000),
                    "timeout": getattr(llm, "timeout", self.max_execution_time),
                    "callbacks": getattr(llm, "callbacks", None),
                    "temperature": getattr(llm, "temperature", None),
                    "logprobs": getattr(llm, "logprobs", None),
                    "api_key": getattr(llm, "api_key", os.environ.get("LITELLM_API_KEY", None)),
                    "base_url": getattr(llm, "base_url", None),
                }
                config = llm_params.update(self.llm_config) if self.llm_config else llm_params
                return self._set_llm_params(llm=llm_obj, config=config)


    def _set_llm_params(self, llm: LLM, config: Dict[str, Any] = None) -> LLM:
        """
        Add valid params to the LLM object.
        """

        import litellm
        from versionhq.llm.llm_vars import PARAMS

        valid_config = {k: v for k, v in config.items() if v} if config else {}

        if valid_config:
            valid_keys = list()
            try:
                valid_keys = litellm.get_supported_openai_params(model=llm.model, custom_llm_provider=self.endpoint_provider, request_type="chat_completion")
                if not valid_keys:
                    valid_keys = PARAMS.get("common")
            except:
                valid_keys = PARAMS.get("common")

            valid_keys += PARAMS.get("litellm")

            for key in valid_keys:
                if key in valid_config and valid_config[key]:
                    val = valid_config[key]
                    if [key == k for k, v in LLM.model_fields.items()]:
                        setattr(llm, key, val)
                    else:
                        llm.other_valid_config.update({ key: val})


        llm.timeout = self.max_execution_time if llm.timeout is None else llm.timeout
        llm.max_tokens = self.max_tokens if self.max_tokens else llm.max_tokens

        if llm.provider is None:
            provider_name = llm.model.split("/")[0]
            valid_provider = provider_name if provider_name in PROVIDERS else None
            llm.provider = valid_provider

        if self.callbacks:
            llm.callbacks = self.callbacks
            llm._set_callbacks(llm.callbacks)

        if self.respect_context_window == False:
            llm.context_window_size = DEFAULT_CONTEXT_WINDOW_SIZE

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
                if isinstance(item, Tool) or isinstance(item, ToolSet):
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
        from versionhq.knowledge.source import BaseKnowledgeSource, StringKnowledgeSource, TextFileKnowledgeSource, CSVKnowledgeSource, ExcelKnowledgeSource, JSONKnowledgeSource
        from versionhq.knowledge.source_docling import DoclingSource

        if self.knowledge_sources:
            try:
                collection_name = f"{self.role.replace(' ', '_')}"
                knowledge_sources = []
                docling_fp, txt_fp, json_fp, excel_fp, csv_fp, pdf_fp = [], [], [], [], [], []
                str_cont = ""

                for item in self.knowledge_sources:
                    if isinstance(item, BaseKnowledgeSource):
                        knowledge_sources.append(item)

                    elif isinstance(item, str) and "http" in item:
                        docling_fp.append(item)

                    elif isinstance(item, str):
                        match  os.path.splitext(item)[1]:
                            case ".txt": txt_fp.append(item)
                            case ".json": json_fp.append(item)
                            case ".xls" | ".xlsx": excel_fp.append(item)
                            case ".pdf": pdf_fp.append(item)
                            case ".csv": csv_fp.append(item)
                            case _: str_cont += str(item)

                    else:
                        str_cont += str(item)

                if docling_fp: knowledge_sources.append(DoclingSource(file_paths=docling_fp))
                if str_cont: knowledge_sources.append(StringKnowledgeSource(content=str_cont))
                if txt_fp: knowledge_sources.append(TextFileKnowledgeSource(file_paths=txt_fp))
                if csv_fp: knowledge_sources.append(CSVKnowledgeSource(file_path=csv_fp))
                if excel_fp: knowledge_sources.append(ExcelKnowledgeSource(file_path=excel_fp))
                if json_fp: knowledge_sources.append(JSONKnowledgeSource(file_paths=json_fp))

                self._knowledge = Knowledge(sources=knowledge_sources, embedder_config=self.embedder_config, collection_name=collection_name)

            except:
                self._logger.log(level="warning", message="We cannot find the format for the source. Add BaseKnowledgeSource objects instead.", color="yellow")

        return self


    @model_validator(mode="after")
    def set_up_memory(self) -> Self:
        """
        Set up memories: stm, ltm, and um
        """

        # if self.use_memory == True:
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


    def update_llm(self, llm: Any = None, llm_config: Optional[Dict[str, Any]] = None) -> Self:
        """
        Update llm and llm_config of the exsiting agent. (Other conditions will remain the same.)
        """

        if not llm and not llm_config:
            self._logger.log(level="error", message="Missing llm or llm_config values to update", color="red")
            pass

        self.llm = llm
        if llm_config:
            if self.llm_config:
                self.llm_config.update(llm_config)
            else:
                self.llm_config = llm_config

        return self.set_up_llm()


    def update(self, **kwargs) -> Self:
        """
        Update the existing agent. Address variables that require runnning set_up_x methods first, then update remaining variables.
        """

        if not kwargs:
            self._logger.log(level="error", message="Missing values to update", color="red")
            return self

        for k, v in kwargs.items():
            match k:
                case "tools":
                    self.tools = kwargs.get(k, self.tools)
                    self.set_up_tools()

                case "role" | "goal":
                    self.role = kwargs.get("role", self.role)
                    self.goal = kwargs.get("goal", self.goal)
                    if not self.backstory:
                        self.set_up_backstory()

                    if self.backstory:
                        self.backstory += f"new role: {self.role}, new goal: {self.goal}"

                case "max_rpm":
                    self.max_rpm = kwargs.get(k, self.max_rpm)
                    self.set_up_rpm()

                case "knowledge_sources":
                    self.knowledge_sources = kwargs.get("knowledge_sources", self.knowledge_sources)
                    self.set_up_knowledge()

                case "use_memory" | "memory_config":
                    self.use_memory = kwargs.get("use_memory", self.use_memory)
                    self.memory_config = kwargs.get("memory_config", self.memory_config)
                    self.set_up_memory()

                case "llm" | "llm_config":
                    self.llm = kwargs.get("llm", self.llm)
                    self.llm_config = kwargs.get("llm_config", self.llm_config)
                    self.update_llm(llm=self.llm, llm_config=self.llm_config)

                case _:
                    try:
                        setattr(self, k, v)
                    except Exception as e:
                        self._logger.log(level="error", message=f"Failed to update the key: {k} We'll skip. Error: {str(e)}", color="red")
                        pass

        return self



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
                raw_response = self.function_calling_llm.call(messages=messages, tools=tools, tool_res_as_final=True)
                task.tokens = self.function_calling_llm._tokens
            else:
                raw_response = self.llm.call(messages=messages, response_format=response_format, tools=tools)
                task.tokens = self.llm._tokens

            task_execution_counter += 1
            self._logger.log(level="info", message=f"Agent response: {raw_response}", color="green")
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
                self._logger.log(level="info", message=f"Agent #{task_execution_counter} response: {raw_response}", color="green")
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
        tools: Optional[List[InstanceOf[Tool | ToolSet] | Type[Tool]]] = task_tools + self.tools if task.can_use_agent_tools else task_tools

        if self.max_rpm and self._rpm_controller:
            self._rpm_controller._reset_request_count()

        task_prompt = task.prompt(model_provider=self.llm.provider)
        if context is not task.prompt_context:
            task_prompt += context

        if self._knowledge:
            agent_knowledge = self._knowledge.query(query=[task_prompt,], limit=5)
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


        ## comment out for now
        # if self.network and self.network._train:
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
        return f"Agent(role={self.role}, goal={self.goal}"
