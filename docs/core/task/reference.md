## Class `Task`

### Variables

| <div style="width:160px">**Variable**</div> | **Data Type**       | **Default**               | **Description**                   |
| :---                      | :---                                  | :---                      | :---                              |
| **`id`**                  | UUID                                  | uuid.uuid4()              | Stores task `id` as an identifier. |
| **`name`**                | Optional[str]                         | None                      | Stores a task name (Inherited as `node` identifier if the task is dependent) |
| **`description`**         | str                                   | None                      | Required field to store a concise task description |
| **`pydantic_output`**     | Optional[Type[BaseModel]]             | None                      | Stores pydantic custom output class for structured response |
| **`response_fields`**     | Optional[List[ResponseField]]         | list()                    | Stores JSON formats for stuructured response |
| **`tools`**               | Optional[List[BaseTool \| ToolSet]]   | None                      | Stores tools to be called when the agent executes the task. |
| **`can_use_agent_tools`** | bool                                  | True                      | Whether to use the agent tools |
| **`tool_res_as_final`**   | bool                                  | False                     | Whether to make a tool output as a final response from the agent |
| **`image`**               | Optional[str]                         | None                      | Stores an absolute file path or URL to the image file in string |
| **`file`**                | Optional[str]                         | None                      | Stores an absolute file path or URL to the file in string |
| **`audio`**               | Optional[str]                         | None                      | Stores an absolute file path or URL to the audio file in string |
| **`execution_type`**      | TaskExecutionType                     | TaskExecutionType.SYNC    | Sync or async execution |
| **`allow_delegation`**    | bool                                  | False                     | Whether to allow the agent to delegate the task to another agent |
| **`callback`**            | Optional[Callable]                    | None                      | Callback function to be executed after LLM calling |
| **`callback_kwargs`**     | Optional[Dict[str, Any]]              | dict()                    | Args for the callback function (if any)|
| **`should_evaluate`**     | bool                                  | False                     | Whether to evaluate the task output using eval criteria |
| **`eval_criteria`**       | Optional[List[str]]                   | list()                    | Evaluation criteria given by the human client |
| **`fsls`**                | Optional[List[str]]                   | None                      | Examples of competitive and/or weak responses |
| **`processed_agents`**    | Set[str]                              | set()                     | Stores keys of agents that executed the task |
| **`output`**              | Optional[TaskOutput]                  | None                      | Stores `TaskOutput` object after the execution |


### Class Methods

| <div style="width:160px">**Method**</div> |  <div style="width:300px">**Params**</div> | **Returns** | **Description** |
| :---               | :---  | :--- | :--- |
| **`execute`**  | <p>type: TaskExecutionType = None<br>agent: Optional["vhq.Agent"] = None<br>context: Optional[Any] = None</p> | InstanceOf[`TaskOutput`] or None (error) |  A main method to handle task execution. Auto-build an agent when the agent is not given. |


### Properties

| <div style="width:160px">**Property**</div>   | **Data_Type** | **Description**                                                           |
| :---                                          | :---          | :---                                                                      |
| **`key`**                                     | str           | Returns task key based on its description and output format.              |
| **`summary`**                                 | str           | Returns a summary of the task based on its id, description and tools.     |

<hr>

## Class `ResponseField`

### Variables

| <div style="width:160px">**Variable**</div> | **Data Type**   | **Default**   | **Description**                                       |
| :---              | :---                                      | :---          | :---                                                  |
| **`title`**       | str                                       | None          | Stores a field title.                                 |
| **`data_type`**   | Type                                      | None          | Stores data type of the response.                     |
| **`items`**       | Type                                      | None          | Stores data type of items in the list. `None` when `data_type` is not list. |
| **`properties`**  | List[`ResponseField`]                     | None          | Stores properties in a list of `ResponseFormat` objects when the `data_type` is dict. |
| **`nullable`**    | bool                                      | False         | If the field is nullable.                             |
| **`config`**      | Dict[str, Any]                            | None          | Stores other configs passed to response schema.       |


<hr>

## Class `TaskOutput`

### Variables

| <div style="width:160px">**Variable**</div>       | **Data Type** | **Default**   |  **Description**                                      |
| :---                      | :---                                  | :---          | :---                                                  |
| **`task_id`**             | UUID                                  | uuid.uuid4()  | Stores task `id` as an identifier.                    |
| **`raw`**                 | str                                   | None          | Stores response in plane text format. `None` or `""` when the model returned errors.|
| **`json_dict`**           | Dict[str, Any]                        | None          | Stores response in JSON serializable dictionary. When the system failed formatting or executing tasks without response_fields, `{ output: <res.raw> }` will be returned. |
| **`pydantic`**            | Type[`BaseModel`]                     | None          | Populates and stores Pydantic class object defined in the `pydantic_output` field if given. |
| **`tool_output`**         | Optional[Any]                         | None          | Stores results from the tools of the task or agents ONLY when `tool_res_as_final` set as `True`. |
| **`callback_output`**     | Optional[Any]                         | None          | Stores results from callback functions if any. |
| **`latency`**             | Optional[float]                       | None          | Stores job latency in milseconds. |
| **`evaluation`**          | Optional[InstanceOf[`Evaluation`]]    | None          | Stores overall evaluations and usage of the task output. |


### Class Methods

| <div style="width:160px">**Method**</div> | **Params** | **Returns** | **Description** |
| :---                   | :---  | :--- | :--- |
| **`evaluate`**         | task: InstanceOf[`Task`]  | InstanceOf[`Evaluation`]  | Evaluates task output based on the criteria |


### Properties

| <div style="width:160px">**Property**</div>   | **Data_Type** | **Description**                                               |
| :---                                          | :---          | :---                                                          |
| **`aggregate_score`**                         | float         | Calucurates weighted average eval scores of the task output.  |
| **`json_string`**                             | str           | Returns `json_dict` in string format.                         |


<hr>

## Class `Evaluation`

### Variables

| <div style="width:160px">**Variable**</div> | **Data Type**   | **Default**   |  **Description**                                  |
| :---              | :---                                      | :---          | :---                                              |
| **`items`**       | List[InstanceOf[EvaluationItem]]          | list()        | Stores evaluation items.                          |
| **`eval_by`**     | Optional[InstanceOf[Agent]]               | None          | Stores an agent assigned to evaluate the output.  |


### Properties

| <div style="width:160px">**Property**</div>   | **Data_Type**   | **Description**                                               |
| :---                                          | :---          | :---                                                          |
| **`aggregate_score`**                         | float         | Calucurates weighted average eval scores of the task output.  |
| **`suggestion_summary`**                      | str           | Returns summary of the suggestions.                           |


<hr>

## SubClass `EvaluationItem`

### Variables

| <div style="width:160px">**Variable**</div>   | **Data Type** | **Default**   |  **Description**                                              |
| :---                                          | :---          | :---          | :---                                                          |
| **`criteria`**                                | str           | None          | Stores evaluation criteria given by the client.               |
| **`suggestion`**                              | str           | None          | Stores suggestion on improvement from the evaluator agent.    |
| **`score`**                                   | float         | None          | Stores the score on a 0 to 1 scale.                           |
| **`weight`**                                  | int           | None          | Stores the weight (importance of the criteria) at any scale.  |
