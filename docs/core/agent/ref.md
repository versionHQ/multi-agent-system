## Variables

| <div style="width:160px">**Variable**</div> | **Data Type** | **Default** | **Nullable** | **Description** |
| :---               | :---  | :--- | :--- | :--- |
| **`id`**   | UUID4 | uuid.uuid4() | False | Stores auto-generated ID as identifier. Not editable. |
| **`role`**   | str | None | False | Stores a role of the agent. |
| **`goal`**   | str | None | False | Stores a goal of the agent. |
| **`backstory`**  | str | None | True | Stores backstory of the agent. Utilized as system prompt. |
| **`tools`**  | List[InstanceOf[`Tool` \| `ToolSet`] \| Type[`Tool`]] | None | True | Stores tools to be used when executing a task. |
| **`knowledge_sources`**  | List[`BaseKnowledgeSource` \| Any] | None | True | Stores knowledge sources in text, file path, or url. |
| **`embedder_config`** | Dict[str, Any] | None | True | Stores embedding configuration for storing knowledge sources. |
| **`with_memory`** | bool | False | - | Whether to store tasks and results in memory. |
| **`memory_config`** |  Dict[str, Any] | None | True | Stores configuration of the memory. |
| **`short_term_memory`** | InstanceOf[`ShortTermMemory`] | None | True | Stores `ShortTermMemory` object. |
| **`long_term_memory`** | InstanceOf[`LongTermMemory`] | None | True | Stores `LongTermMemory` object. |
| **`user_memory`** | InstanceOf[`UserMemory`] | None | True | Stores `UserMemory` object. |
| **`use_developer_prompt`** |  bool | True | - | Whether to use the system (developer) prompt when calling the model. |
| **`developer_promt_template`** | str | None | True | File path to the prompt template. |
| **`user_promt_template`** | str | None | True | File path to the prompt template. |
| **`networks`** | List[Any] | list() | True | Stores a list of agent networks that the agent belongs to. |
| **`allow_delegation`** | bool | False | - | Whether the agent can delegate assinged tasks to another agent. |
| **`max_retry_limit`** | int | 2 | - | Maximum number of retries when the task execution failed. |
| **`maxit`** | int | 25 | - | Maximum number of total optimization loops conducted when an error occues during the task execution. |
| **`callbacks`** | List[Callabale] | None | True | Stores a list of callback functions that must be called after every task execution completed.|
| **`llm`** | str \| InstanceOf[`LLM`] \| Dict[str, Any] | None | False | Stores the main model that the agent runs on. |
| **`func_calling_llm`** | str \| InstanceOf[`LLM`] \| Dict[str, Any] | None | False | Stores the function calling model that the agent runs on. |
| **`respect_context_window`** | bool | True | - | Whether to follow the main model's maximum context window size. |
| **`max_execution_time`** | int | None | True | Stores maximum execution time in seconds. |
| **`max_rpm`** | int | None | True | Stores maximum number of requests per minute. |
| **`llm_config`** | Dict[str, Any] | None | True | Stores configuration of `LLM` object. |
| **`config`** | Dict[str, Any] | None | True | Stores model config. |


## Class Methods

| <div style="width:120px">**Method**</div> | **Params** | **Returns** | **Description** |
| :---               | :---  | :--- | :--- |
| **`update`** | **kwargs: Any  | Self | Updates agents with given kwargs. Invalid keys will be ignored. |
| **`start`** | context: Any = None <br> tool_res_as_final: bool = False  | `TaskOutput` \| None  | Starts to operate the agent. |
| **`execute_task`** | task: [Task] <br> context: Any = None <br> task_tools: Optional[List[Tool \| ToolSet]] = list() | str | Returns response from the model in plane text format. |
