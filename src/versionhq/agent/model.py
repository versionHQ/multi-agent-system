import os
import uuid
from typing import Any, Dict, List, Optional, TypeVar, Callable, Type, Tuple
from typing_extensions import Self
from dotenv import load_dotenv

from pydantic import UUID4, BaseModel, Field, InstanceOf, PrivateAttr, model_validator, field_validator
from pydantic_core import PydanticCustomError

from versionhq.agent.rpm_controller import RPMController
from versionhq.tool.model import Tool, ToolSet, BaseTool
from versionhq.knowledge.model import BaseKnowledgeSource, Knowledge
from versionhq.memory.model import ShortTermMemory, LongTermMemory, UserMemory
from versionhq._utils import Logger, process_config, is_valid_url, ErrorType, UsageMetrics


load_dotenv(override=True)
T = TypeVar("T", bound="Agent")


# @track_agent()
class Agent(BaseModel):
    """
    A Pydantic class to store an agent object. Agents must have `role` and `goal` to start.
    """

    __hash__ = object.__hash__
    _rpm_controller: Optional[RPMController] = PrivateAttr(default=None)
    _request_within_rpm_limit: Any = PrivateAttr(default=None)
    _times_executed: int = PrivateAttr(default=0)
    _logger_config: Dict[str, Any] = PrivateAttr(default=dict(verbose=True, info_file_save=True))

    api_key: Optional[str] = Field(default=None)
    self_learn: bool = Field(default=False)
    config: Optional[Dict[str, Any]] = Field(default=None, exclude=True, description="values to add to the Agent class")

    id: UUID4 = Field(default_factory=uuid.uuid4, frozen=True)
    role: str = Field(description="required. agent's role")
    goal: Optional[str] = Field(default=None)
    backstory: Optional[str] = Field(default=None, description="developer prompt to the llm")
    skills: Optional[List[str]] = Field(default_factory=list, description="list up the agent's tangible skills in natural language")
    tools: Optional[List[Any]] = Field(default_factory=list)

    # knowledge
    knowledge_sources: Optional[List[BaseKnowledgeSource | Any]] = Field(default=None)
    embedder_config: Optional[Dict[str, Any]] = Field(default=None, description="embedder configuration for knowledge sources")
    _knowledge: Optional[Knowledge] = PrivateAttr(default=None)

    # memory
    with_memory: bool = Field(default=False, description="whether to use memories during the task execution")
    memory_config: Optional[Dict[str, Any]] = Field(default=None, description="memory config. needs to store user_id for UserMemory to work")
    short_term_memory: Optional[InstanceOf[ShortTermMemory]] = Field(default=None)
    long_term_memory: Optional[InstanceOf[LongTermMemory]] = Field(default=None)
    user_memory: Optional[InstanceOf[UserMemory]] = Field(default=None)

    # prompting
    use_developer_prompt: Optional[bool] = Field(default=True, description="Use developer prompt when calling the llm")
    developer_propmt_template: Optional[str] = Field(default=None, description="abs. file path to developer prompt template")
    user_prompt_template: Optional[str] = Field(default=None, description="abs. file path to user prompt template")

    # task execution rules
    networks: Optional[List[Any]] = Field(default_factory=list, description="store a list of agent networks that the agent belongs to as a member")
    allow_delegation: bool = Field(default=False, description="whether to delegate the task to another agent")
    max_retry_limit: int = Field(default=2, description="max. number of task retries when an error occurs")
    maxit: Optional[int] = Field(default=25, description="max. number of total optimization loops conducted when an error occurs")
    callbacks: Optional[List[Callable]] = Field(default_factory=list, description="callback functions to execute after any task execution")

    # llm settings cascaded to the LLM model
    llm: Any = Field(default=None, description="store LLM object")
    func_calling_llm: Any = Field(default=None, description="store LLM object")
    respect_context_window: bool = Field(default=True, description="keep messages under the context window size")
    max_execution_time: Optional[int] = Field(default=None, description="max. task execution time in seconds")
    max_rpm: Optional[int] = Field(default=None, description="max. number of requests per minute")
    llm_config: Optional[Dict[str, Any]] = Field(default=None, description="other llm config cascaded to the LLM class")

    # # cache, error, ops handling
    # formatting_errors: int = Field(default=0, description="number of formatting errors.")

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
        required_fields = ["role",]
        for field in required_fields:
            if getattr(self, field) is None:
                raise ValueError(f"{field} must be provided either directly or through config")
        return self


    @model_validator(mode="after")
    def set_up_llm(self) -> Self:
        """
        Set up `llm` and `func_calling_llm` as valid LLM objects using the given kwargs.
        """
        from versionhq.llm.model import LLM, DEFAULT_MODEL_NAME

        self.llm = self._convert_to_llm_object(llm=self.llm)

        func_calling_llm = self.func_calling_llm if self.func_calling_llm else self.llm if self.llm else None
        func_calling_llm = self._convert_to_llm_object(llm=func_calling_llm)
        if func_calling_llm._supports_function_calling():
            self.func_calling_llm = func_calling_llm
        elif self.llm._supports_function_calling():
            self.func_calling_llm = self.llm
        else:
            self.func_calling_llm = self._convert_to_llm_object(llm=LLM(model=DEFAULT_MODEL_NAME))
        return self


    @model_validator(mode="after")
    def set_up_tools(self) -> Self:
        """
        Similar to the LLM set up, when the agent has tools, we will declare them using the Tool class.
        """
        from versionhq.tool.rag_tool import RagTool
        from versionhq.tool.gpt.web_search import GPTToolWebSearch
        from versionhq.tool.gpt.file_search import GPTToolFileSearch
        from versionhq.tool.gpt.cua import GPTToolCUA

        if not self.tools:
            return self

        tool_list = []
        for item in self.tools:
            match item:
                case RagTool() | BaseTool() | GPTToolCUA() | GPTToolFileSearch() | GPTToolWebSearch():
                    tool_list.append(item)

                case Tool():
                    if item.func is not None:
                        tool_list.append(item)

                case ToolSet():
                    if item.tool and item.tool.func is not None:
                        tool_list.append(item)

                case dict():
                    if "func" in item:
                        tool = Tool(func=item["func"])
                        for k, v in item.items():
                            if k in Tool.model_fields.keys() and k != "func" and  v is not None:
                                setattr(tool, k, v)
                        tool_list.append(tool)

                case func if callable(func):
                    tool = Tool(func=item)
                    tool_list.append(tool)

                case _:
                    if hasattr(item, "__base__") and (item.__base__ == BaseTool or item.__base__ == RagTool or item.__base__ == Tool):
                        tool_list.append(item)
                    else:
                        Logger(**self._logger_config, filename=self.key).log(level="error", message=f"Tool {str(item)} is missing a function.", color="red")
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
            skills = ", ".join([item for item in self.skills]) if self.skills else ""
            tools = ", ".join([item.name for item in self.tools if hasattr(item, "name") and item.name is not None]) if self.tools else ""
            role = self.role.lower()
            goal = self.goal.lower() if self.goal else ""

            if self.tools or self.skills:
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
                collection_name = self.key
                knowledge_sources = []
                docling_fp, txt_fp, json_fp, excel_fp, csv_fp, pdf_fp = [], [], [], [], [], []
                str_cont = ""

                for item in self.knowledge_sources:
                    if isinstance(item, BaseKnowledgeSource):
                        knowledge_sources.append(item)

                    elif isinstance(item, str) and "http" in item and is_valid_url(url=item) == True:
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

            except Exception as e:
                Logger(**self._logger_config, filename=self.key).log(level="warning", message=f"We cannot find the format for the source. Add BaseKnowledgeSource objects instead. {str(e)}", color="yellow")

        return self


    @model_validator(mode="after")
    def set_up_memory(self) -> Self:
        """
        Set up memories: stm, ltm, and um
        """

        # if self.with_memory == True:
        self.long_term_memory = self.long_term_memory if self.long_term_memory else LongTermMemory()
        self.short_term_memory = self.short_term_memory if self.short_term_memory else ShortTermMemory(agent=self, embedder_config=self.embedder_config)

        if hasattr(self, "memory_config") and self.memory_config is not None:
            user_id = self.memory_config.get("user_id", None)
            if user_id:
                self.user_memory = self.user_memory if self.user_memory else UserMemory(agent=self, user_id=user_id)
        else:
            self.user_memory = None

        return self


    def _convert_to_llm_object(self, llm: Any = None): # returns LLM object
        """
        Convert the given value to LLM object.
        When `llm` is dict or self.llm_config is not None, add these values to the LLM object after validating them.
        """
        from versionhq.llm.model import LLM, DEFAULT_MODEL_NAME

        llm = llm if llm else self.llm if self.llm else DEFAULT_MODEL_NAME

        if not llm:
            pass

        match llm:
            case LLM():
                return self._set_llm_params(llm_obj=llm, config=self.llm_config)

            case str():
                llm = LLM(model=llm)
                return self._set_llm_params(llm_obj=llm, config=self.llm_config)

            case dict():
                model_name = llm.pop("model_name", llm.pop("deployment_name", str(llm)))
                llm_obj = LLM(model=model_name if model_name else DEFAULT_MODEL_NAME)
                config = llm.update(self.llm_config) if self.llm_config else llm
                return self._set_llm_params(llm_obj, config=config)

            case _:
                model_name = (getattr(self.llm, "model_name") or getattr(self.llm, "deployment_name") or str(self.llm))
                llm = LLM(model=model_name if model_name else DEFAULT_MODEL_NAME)
                llm_params = {
                    "timeout": getattr(llm, "timeout", self.max_execution_time),
                    "callbacks": getattr(llm, "callbacks", None),
                    "llm_config": getattr(llm, "llm_config", None),
                    "base_url": getattr(llm, "base_url", None),
                }
                config = llm_params.update(self.llm_config) if self.llm_config else llm_params
                return self._set_llm_params(llm_obj=llm, config=config)


    def _set_llm_params(self, llm_obj, config: Dict[str, Any] = None):  # returns LLM object
        """
        Add valid params to the LLM object.
        """
        from versionhq.llm.model import LLM, DEFAULT_CONTEXT_WINDOW_SIZE
        from versionhq.llm.llm_vars import PROVIDERS

        llm = llm_obj if isinstance(llm_obj, LLM) else None

        if not llm: return

        if llm.provider is None:
            provider_name = llm.model.split("/")[0]
            valid_provider = provider_name if provider_name in PROVIDERS else None
            llm.provider = valid_provider

        if self.callbacks:
            llm.callbacks = self.callbacks
            llm._set_callbacks(llm.callbacks)

        if self.respect_context_window == False:
            llm.context_window_size = DEFAULT_CONTEXT_WINDOW_SIZE

        llm.timeout = self.max_execution_time if llm.timeout is None else llm.timeout

        if config:
            llm.llm_config = {k: v for k, v in config.items() if v or v == False}
            llm.setup_config()

        return llm


    def _train(self) -> Self:
        """
        Fine-tuned the base model using OpenAI train framework.
        """
        from versionhq.llm.model import LLM

        if not isinstance(self.llm, LLM):
            pass


    def _invoke(
        self,
        messages: List[Dict[str, str]] = None,
        response_format: Optional[Dict[str, Any]] = None,
        tools: Optional[List[InstanceOf[Tool]| InstanceOf[ToolSet] | Type[Tool]]] = None,
        tool_res_as_final: bool = False,
        file: str = None, # absolute path to the content file (for multimodal use)
        ) -> Tuple[str, UsageMetrics]:
        """Calls LLM."""

        task_execution_counter = 0
        iterations = 0
        raw_response = None
        usage = UsageMetrics()

        try:
            if self._rpm_controller and self.max_rpm:
                self._rpm_controller.check_or_wait()

            Logger(**self._logger_config, filename=self.key).log(level="info", message=f"Messages sent to the model: {messages}", color="blue")

            if tool_res_as_final:
                raw_response = self.func_calling_llm.call(messages=messages, tools=tools, tool_res_as_final=True, file=file)
                usage.record_token_usage(*self.func_calling_llm._usages)
            else:
                raw_response = self.llm.call(messages=messages, response_format=response_format, tools=tools, file=file)
                usage.record_token_usage(*self.llm._usages)

            task_execution_counter += 1
            Logger(**self._logger_config, filename=self.key).log(level="info", message=f"Agent response: {raw_response}", color="green")
            return raw_response, usage

        except Exception as e:
            usage.record_errors(type=ErrorType.API)
            Logger(**self._logger_config, filename=self.key).log(level="error", message=f"An error occured. The agent will retry: {str(e)}", color="red")

            while not raw_response and task_execution_counter <= self.max_retry_limit:
                while (not raw_response or raw_response == "" or raw_response is None) and iterations < self.maxit:
                    if self.max_rpm and self._rpm_controller:
                        self._rpm_controller.check_or_wait()

                    raw_response = self.llm.call(messages=messages, response_format=response_format, tools=tools)
                    usage.record_token_usage(*self.llm._usages)
                    iterations += 1

                task_execution_counter += 1
                Logger(**self._logger_config, filename=self.key).log(level="info", message=f"Agent #{task_execution_counter} response: {raw_response}", color="green")
                return raw_response, usage

            if not raw_response:
                Logger(**self._logger_config, filename=self.key).log(level="error", message="Received None or empty response from the model", color="red")
                raise ValueError("Invalid response from LLM call - None or empty.")


    def _update_llm(self, llm: Any = None, llm_config: Optional[Dict[str, Any]] = None) -> Self:
        """
        Updates llm and llm_config of the exsiting agent. (Other conditions will remain the same.)
        """

        if not llm and not llm_config:
            Logger(**self._logger_config, filename=self.key).log(level="error", message="Missing llm or llm_config values to update", color="red")
            pass

        self.llm = llm
        if llm_config:
            if self.llm_config:
                self.llm_config.update(llm_config)
            else:
                self.llm_config = llm_config

        return self.set_up_llm()


    def _sort_tools(self, task = None) -> Tuple[List[Any], List[Any], List[Any]]:
        """Sorts agent and task tools by class."""

        from versionhq.tool.rag_tool import RagTool
        from versionhq.tool.gpt.web_search import GPTToolWebSearch
        from versionhq.tool.gpt.file_search import GPTToolFileSearch
        from versionhq.tool.gpt.cua import GPTToolCUA

        all_tools = []
        if task: all_tools = task.tools + self.tools if task.can_use_agent_tools else task.tools
        else: all_tools = self.tools

        rag_tools, gpt_tools, tools = [], [], []
        if all_tools:
            for item in all_tools:
                match item:
                    case RagTool():
                        rag_tools.append(item)

                    case GPTToolCUA() | GPTToolFileSearch() | GPTToolWebSearch():
                        gpt_tools.append(item)

                    case Tool() | BaseTool() | ToolSet():
                        tools.append(item)

        return rag_tools, gpt_tools, tools


    def _handle_gpt_tools(self, gpt_tools: list[Any] = None) -> Any: # TaskOutput or None
        """Generates k, v pairs from multiple GPT tool results and stores them in TaskOutput class."""

        from versionhq.task.model import TaskOutput
        from versionhq._utils import UsageMetrics

        if not gpt_tools:
            return None

        tool_res = dict()
        annotation_set = dict()
        total_usage = UsageMetrics()

        for i, item in enumerate(gpt_tools):
            raw, annotations, usage = item.run()
            tool_res.update({ str(i): raw })

            if annotations:
                annotation_set.update({ str(i): annotations })
            total_usage.aggregate(metrics=usage)

        res = TaskOutput(raw=str(tool_res), tool_output=tool_res, usage=total_usage, annotations=annotation_set)
        return res


    def update(self, **kwargs) -> Self:
        """
        Update the existing agent. Address variables that require runnning set_up_x methods first, then update remaining variables.
        """

        if not kwargs:
            Logger(**self._logger_config, filename=self.key).log(level="error", message="Missing values to update", color="red")
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

                case "with_memory" | "memory_config":
                    self.with_memory = kwargs.get("with_memory", self.with_memory)
                    self.memory_config = kwargs.get("memory_config", self.memory_config)
                    self.set_up_memory()

                case "llm" | "llm_config":
                    self.llm = kwargs.get("llm", self.llm)
                    self.llm_config = kwargs.get("llm_config", self.llm_config)
                    self._update_llm(llm=self.llm, llm_config=self.llm_config)

                case _:
                    try:
                        setattr(self, k, v)
                    except Exception as e:
                        Logger(**self._logger_config, filename=self.key).log(level="error", message=f"Failed to update the field: {k} We'll skip it. Error: {str(e)}", color="red")
                        pass

        return self


    def start(
            self,
            context: Any = None,
            tool_res_as_final: bool = False,
            image: str = None,
            file: str = None,
            audio: str = None
        ) -> Any:
        """
        Defines and executes a task, then returns TaskOutput object with the generated task.
        """

        from versionhq.task.model import Task

        if not self.role:
            return None, None

        _, gpt_tools, _ = self._sort_tools()

        if gpt_tools and tool_res_as_final == True:
            res = self._handle_gpt_tools(gpt_tools=gpt_tools)
            return res

        class Output(BaseModel):
            result: str
            steps: list[str]

        task = Task(
            description=f"Generate a simple result in a sentence to achieve the goal: {self.goal if self.goal else self.role}. If needed, list up necessary steps in concise manner.",
            response_schema=Output,
            tool_res_as_final=tool_res_as_final,
            image=image, #REFINEME - query memory/knowledge or self create
            file=file,
            audio=audio,
            can_use_agent_tools=True if self.tools else False,
        )
        res = task.execute(agent=self, context=context)
        return res


    def execute_task(self, task, context: Optional[Any] = None) -> Tuple[str, str, Any, UsageMetrics]:
        """Handling task execution."""

        from versionhq._prompt.model import Prompt
        from versionhq.task.model import Task

        task: InstanceOf[Task] = task
        rag_tools, gpt_tools, tools = self._sort_tools(task=task)
        raw_response = ""
        user_prompt, dev_prompt = "", ""
        usage = UsageMetrics(id=task.id)
        file = task.audio if task.is_multimodal and task.audio else task.image if task.is_multimodal and task.image else task.file if task.is_multimodal and task.file else None

        if self.max_rpm and self._rpm_controller:
            self._rpm_controller._reset_request_count()

        if task.tool_res_as_final == True and gpt_tools:
            self._times_executed += 1
            res = self._handle_gpt_tools(gpt_tools=gpt_tools)
            return user_prompt, dev_prompt, res, res.usage

        user_prompt, dev_prompt, messages = Prompt(task=task, agent=self, context=context).format_core(rag_tools=rag_tools, gpt_tools=gpt_tools)

        try:
            self._times_executed += 1
            raw_response, usage = self._invoke(
                messages=messages,
                response_format=task._structure_response_format(model_provider=self.llm.provider),
                tools=tools,
                tool_res_as_final=task.tool_res_as_final,
                file=file,
            )

        except Exception as e:
            self._times_executed += 1
            Logger(**self._logger_config, filename=self.key).log(level="error", message=f"The agent failed to execute the task. Error: {str(e)}", color="red")
            user_prompt, dev_prompt, raw_response, usage = self.execute_task(task, context)

            if self._times_executed > self.max_retry_limit:
                Logger(**self._logger_config, filename=self.key).log(level="error", message=f"Max retry limit has exceeded.", color="red")
                raise e

        if self.max_rpm and self._rpm_controller:
            self._rpm_controller.stop_rpm_counter()

        return user_prompt, dev_prompt, raw_response, usage


    @property
    def key(self):
        """
        A key to identify an agent. Used in storage, logging, and other recodings.
        """
        sanitized_role = self.role.lower().replace(" ", "-").replace("/", "").replace("{", "").replace("}", "").replace("\n", "")[0: 16]
        return f"{str(self.id)}-{sanitized_role}"


    def __repr__(self):
        return f"Agent(role={self.role}, id={str(self.id)}"

    def __str__(self):
        return super().__str__()
