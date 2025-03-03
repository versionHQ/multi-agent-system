## `Task`

### Variables

| <div style="width:160px">**Variable**</div> | **Data Type** | **Default** | **Nullable** | **Description** |
| :---               | :---  | :--- | :--- | :--- |
| **`id`**   | UUID  | uuid.uuid4() | False | Stores task `id` as an identifier. |
| **`name`**       | Optional[str]   | None | True | Stores a task name (Inherited as `node` identifier if the task is dependent) |
| **`description`**       | str   | None | False | Required field to store a concise task description |
| **`pydantic_output`** | Optional[Type[BaseModel]] | None | True | Stores pydantic custom output class for structured response |
| **`response_fields`** | Optional[List[ResponseField]]  | list() | True | Stores JSON formats for stuructured response |
| **`tools`** |  Optional[List[ToolSet | Tool | Any]] | None | True | Stores tools to be called when the agent executes the task. |
| **`can_use_agent_tools`** |  bool | True | - | Whether to use the agent tools |
| **`tool_res_as_final`** |  bool | False | - | Whether to make the tool response a final response from the agent |
| **`execution_type`** | TaskExecutionType  | TaskExecutionType.SYNC | - | Sync or async execution |
| **`allow_delegation`** | bool  | False | - | Whether to allow the agent to delegate the task to another agent |
| **`callback`** | Optional[Callable] | None | True | Callback function to be executed after LLM calling |
| **`callback_kwargs`** | Optional[Dict[str, Any]] | dict() | True | Args for the callback function (if any)|
| **`should_evaluate`** | bool | False | - | Whether to evaluate the task output using eval criteria |
| **`eval_criteria`** | Optional[List[str]] | list() | True | Evaluation criteria given by the human client |
| **`fsls`** | Optional[List[str]] | None | True | Examples of excellent and weak responses |
| **`processed_agents`** | Set[str] | set() | True | [Ops] Stores roles of the agents executed the task |
| **`tool_errors`** | int | 0 | True | [Ops] Stores number of tool errors |
| **`delegation`** | int | 0 | True | [Ops] Stores number of agent delegations |
| **`output`** | Optional[TaskOutput] | None | True | [Ops] Stores `TaskOutput` object after the execution |


### Class Methods

| <div style="width:120px">**Method**</div> |  <div style="width:300px">**Params**</div> | **Returns** | **Description** |
| :---               | :---  | :--- | :--- |
| **`execute`**  | <p>type: TaskExecutionType = None<br>agent: Optional["vhq.Agent"] = None<br>context: Optional[Any] = None</p> | InstanceOf[`TaskOutput`] or None (error) |  A main method to handle task execution. Auto-build an agent when the agent is not given. |


### Properties

| <div style="width:120px">**Property**</div> | **Returns** | **Description** |
| :---               | :---  | :--- |
| **`key`**   | str | Returns task key based on its description and output format. |
| **`summary`**  | str   | Returns a summary of the task based on its id, description and tools. |

<hr>

## `TaskOutput`

### Variables

| <div style="width:120px">**Variable**</div> | **Data Type** | **Default** | **Nullable** | **Description** |
| :---               | :---  | :--- | :--- | :--- |
| **`task_id`**   | UUID  | uuid.uuid4() | False | Stores task `id` as an identifier. |
| **`raw`**       | str   | None | False | Stores response in plane text format. `None` or `""` when the model returned errors.|
| **`json_dict`** | Dict[str, Any] | None | False | Stores response in JSON serializable dictionary. When the system failed formatting or executing tasks without response_fields, `{ output: <res.raw> }` will be returned. |
| **`pydantic`** | Type[`BaseModel`]  | None | True | Populates and stores Pydantic class object defined in the `pydantic_output` field. `None` if `pydantic_output` is NOT given. |
| **`tool_output`** |  Optional[Any] | None | True | Stores results from the tools of the task or agents ONLY when `tool_res_as_final` set as `True`. |
| **`callback_output`** |  Optional[Any] | None | True | Stores results from callback functions if any. |
| **`latency`** |  Optional[float] | None | True | Stores job latency in milseconds. |
| **`evaluation`** |  Optional[InstanceOf[`Evaluation`]] | None | True | Stores overall evaluations and usage of the task output. |


### Class Methods

| <div style="width:120px">**Method**</div> | **Params** | **Returns** | **Description** |
| :---               | :---  | :--- | :--- |
| **`evaluate`**   | task: InstanceOf[`Task`]  | InstanceOf[`Evaluation`]  | Evaluates task output based on the criteria |


### Property

| <div style="width:120px">**Property**</div> | **Returns** | **Description** |
| :---               | :---  | :--- |
| **`aggregate_score`**   | float | Calucurates weighted average eval scores of the task output. |
| **`json_string`**       | str   | Returns `json_dict` in string format. |


<hr>

## `Evaluation`

### Variables

| <div style="width:120px">**Variable**</div> | **Data Type** | **Default** | **Nullable** | **Description** |
| :---            | :---  | :--- | :--- | :--- |
| **`items`**     | List[InstanceOf[EvaluationItem]]  | list() | - | Stores evaluation items. |
| **`eval_by`**   | Any   | None | True | Stores an agent evaluated the output. |


### Property

| <div style="width:120px">**Property**</div> | **Returns** | **Description** |
| :---               | :---  | :--- |
| **`aggregate_score`**   | float | Calucurates weighted average eval scores of the task output. |
| **`suggestion_summary`**  | str   | Returns summary of the suggestions. |


<hr>

## `EvaluationItem`

### Variables

| <div style="width:120px">**Variable**</div> | **Data Type** | **Default** | **Nullable** | **Description** |
| :---            | :---  | :--- | :--- | :--- |
| **`criteria`**     | str  | None | False | Stores evaluation criteria given by the client. |
| **`suggestion`**   | str   | None | True | Stores suggestion on improvement from the evaluator agent. |
| **`score`**   | float   | None | True | Stores the score on a 0 to 1 scale. |
