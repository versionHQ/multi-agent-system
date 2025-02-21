---
tags:
  - Agent Network
---


## Class `AgentNetwork`

### Variable

| <div style="width:200px">**Variable**</div> | **Data Type** | **Default** | **Nullable** | **Description** |
| :---               | :---  | :--- | :--- | :--- |
| **`id`**   | UUID4 | uuid.uuid4() | False | Stores auto-generated ID as identifier. Not editable. |
| **`name`**   | str | None | True | Stores a name of the network. |
| **`members`**   | List[`Member`] | list() | False | Stores a list of `Member` objects. |
| **`formation`**  | `Formation`| None | True | Stores `Formation` enum. |
| **`should_reform`**  | bool | False | - | Whether to reform the network during the activation. |
| **`network_tasks`**  | List[`Task`] | list() | - | Stores a list of `Task` objects not assigned to any network members. |
| **`prompt_file`** | str | None | True | Stores absolute file path to the prompt file with JSON formatted prompt |
| **`process`** | `TaskHandlingProcess` | TaskHandlingProcess.SEQUENTIAL | - | Enum of the task handling process. |
| **`consent_trigger`** | Callable[..., Any] | None | True | Stores a trigger event (func) for consentual processing. |
| **`pre_launch_callbacks`** | List[Callable[..., Any]] | list() | - | Stores callbacks to run before the network launch. |
| **`post_launch_callbacks`** | List[Callable[..., Any]] | list() | - | Stores callbacks to run after the network launch. |
| **`step_callbacks`** | Callable[..., Any] | None | True | Stores callbacks to run at every step of each member agent takes during the activation. |
| **`cache`** | bool | True | - | Whether to store cache. |
| **`execution_logs`** |  List[Dict[str, Any]] | list() | - | Stores a list of execution logs of all the tasks in the network. |


### Class Methods

| <div style="width:120px">**Method**</div> | **Params** | **Returns** | **Description** |
| :---               | :---  | :--- | :--- |
| **`launch`** | kwargs_pre: Optional[Dict[str, str]] = None <br> kwargs_post: Optional[Dict[str, Any]] = None <br> start_index: int = None |  Tuple[TaskOutput, TaskGraph]: | Core method to launch the network and execute tasks |


### Properties

| <div style="width:120px">**Property**</div> | **Returns** | **Description** |
| :---               | :---  | :--- |
| **`key`** | str | Unique identifier of the netowrk. |
| **`managers`** | List[`Member`] | A list of manager members. |
| **`manager_tasks`** | List[`Task`] | A list of tasks handled by managers. |
| **`tasks`** | List[`Task`] | All the tasks in the network.|
| **`unassigned_member_tasks`** | List[Task] | Unassigned member-level tasks. |


<hr>

## Class `Member`

### Variable

| <div style="width:200px">**Variable**</div> | **Data Type** | **Default** | **Nullable** | **Description** |
| :---               | :---  | :--- | :--- | :--- |
| **`agent`**   | `Agent` | None | True | Agent as a member |
| **`is_manager`**   | bool | False | - | Whether the member is a manager. |
| **`can_share_knowledge`**   | bool | True | - | Whether the member can share its knowledge among the other members in the network. |
| **`can_share_memory`**   | bool | True | - | Whether the member can share its memories among the other members in the network. |
| **`tasks`**  | List[`Task`]| list() | - | Assinged tasks. |


### Properties

| <div style="width:120px">**Property**</div> | **Returns** | **Description** |
| :---               | :---  | :--- |
| **`is_idling`** | bool | Whether it has unprocessed assgined task/s |



<hr>

## Class `Agent`

### Variables

| <div style="width:200px">**Variable**</div> | **Data Type** | **Default** | **Nullable** | **Description** |
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


### Class Methods

| <div style="width:120px">**Method**</div> | **Params** | **Returns** | **Description** |
| :---               | :---  | :--- | :--- |
| **`update`** | **kwargs: Any  | Self | Updates agents with given kwargs. Invalid keys will be ignored. |
| **`start`** | context: Any = None <br> tool_res_as_final: bool = False  | `TaskOutput` \| None  | Starts to operate the agent. |
| **`execute_task`** | task: [Task] <br> context: Any = None <br> task_tools: Optional[List[Tool \| ToolSet]] = list() | str | Returns response from the model in plane text format. |


### Properties

| <div style="width:120px">**Property**</div> | **Returns** | **Description** |
| :---               | :---  | :--- |
| **`key`** | str | Unique identifier of the agent using its ID and sanitized role. |


## ENUM `Formation`

```python
class Formation(str, Enum):
    SOLO = 1
    SUPERVISING = 2
    SQUAD = 3
    RANDOM = 4
    HYBRID = 10
```

## ENUM `TaskHandlingProcess`

```python
class TaskHandlingProcess(str, Enum):
    HIERARCHY = 1
    SEQUENTIAL = 2
    CONSENSUAL = 3
```
