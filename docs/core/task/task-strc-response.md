# Structured Response

By default, agents will generate plane text and JSON outputs, and store them in the `TaskOutput` object.

* Ref. <a href="/core/task/task-output">`TaskOutput`</a> class

But you can choose to generate Pydantic class or specifig JSON object as response.

<hr />

## 1. Pydantic

`[var]`<bold>`pydantic_output: Optional[Type[BaseModel]] = None`</bold>

Add a `custom Pydantic class` as a structured response format to the `pydantic_output` field.

The custom class can accept **one layer of a nested child** as you can see in the following code snippet:

```python
import versionhq as vhq
from pydantic import BaseModel
from typing import Any


# 1. Define Pydantic class with a description (optional), annotations and field names.
class Demo(BaseModel):
    """
    A demo pydantic class to validate the outcome with various nested data types.
    """
    demo_1: int
    demo_2: float
    demo_3: str
    demo_4: bool
    demo_5: list[str]
    demo_6: dict[str, Any]
    demo_nest_1: list[dict[str, Any]] # 1 layer of nested child is ok.
    demo_nest_2: list[list[str]]
    demo_nest_3: dict[str, list[str]]
    demo_nest_4: dict[str, dict[str, Any]]
    # error_1: list[list[dict[str, list[str]]]] # <- Trigger 400 error due to 2+ layers of nested child.
    # error_2: InstanceOf[AnotherPydanticClass] # <- Trigger 400 error due to non-typing annotation.
    # error_3: list[InstanceOf[AnotherPydanticClass]] # <- Trigger 400 error due to non-typing annotation as a nested child.

# 2. Define a task
task = vhq.Task(
    description="generate random output that strictly follows the given format",
    pydantic_output=Demo,
)

# 3. Execute
res = task.execute()

assert isinstance(res, vhq.TaskOutput)
assert res.raw and res.json
assert isinstance(res.raw, str) and isinstance(res.json_dict, dict)
assert [
    getattr(res.pydantic, k) and v.annotation == Demo.model_fields[k].annotation
    for k, v in res.pydantic.model_fields.items()
]
```

<hr />

## 2. JSON

`[var]`<bold>`response_fields: List[InstanceOf[ResponseField]] = None`</bold>

Similar to Pydantic, JSON output structure can be defined by using a list of `ResponseField` objects.

The following code snippet demonstrates how to use `ResponseField` to generate output with a maximum of one level of nesting.

Custom JSON outputs can accept **one layer of a nested child**.

**[NOTES]**

- `demo_response_fields` in the following case is identical to the previous Demo class, except that titles are specified for nested fields.

- Agents generate JSON output by default, whether or not `response_fields` are used.

- However, response_fields are REQUIRED to specify JSON key titles and data types.

```python
import versionhq as vhq

# 1. Define a list of ResponseField objects.
demo_response_fields = [
    # no nesting
    vhq.ResponseField(title="demo_1", data_type=int),
    vhq.ResponseField(title="demo_2", data_type=float),
    vhq.ResponseField(title="demo_3", data_type=str),
    vhq.ResponseField(title="demo_4", data_type=bool),
    vhq.ResponseField(title="demo_5", data_type=list, items=str),
    vhq.ResponseField(
        title="demo_6",
        data_type=dict,
        properties=[vhq.ResponseField(title="demo-item", data_type=str)]
    ),
    # nesting
    vhq.ResponseField(
        title="demo_nest_1",
        data_type=list,
        items=dict,
        properties=([
            vhq.ResponseField(
                title="nest1",
                data_type=dict,
                properties=[vhq.ResponseField(title="nest11", data_type=str)]
            )
        ])
    ),
    vhq.ResponseField(title="demo_nest_2", data_type=list, items=list),
    vhq.ResponseField(title="demo_nest_3", data_type=dict, properties=[
        vhq.ResponseField(title="nest1", data_type=list, items=str)
    ]),
    vhq.ResponseField(title="demo_nest_4", data_type=dict, properties=[
        vhq.ResponseField(
            title="nest1",
            data_type=dict,
            properties=[vhq.ResponseField(title="nest12", data_type=str)]
        )
    ])
]


# 2. Define a task
task = vhq.Task(
    description="Output random values strictly following the data type defined in the given response format.",
    response_fields=demo_response_fields
)


# 3. Execute
res = task.execute()

assert isinstance(res, vhq.TaskOutput) and res.task_id is task.id
assert res.raw and res.json and res.pydantic is None
assert [v and type(v) == task.response_fields[i].data_type for i, (k, v) in enumerate(res.json_dict.items())]
```

* Ref. <a href="/core/task/response-field">`ResponseField`</a> class

<hr />

**Structuring reponse format**

- Higlhy recommends assigning agents optimized for `gemini-x` or `gpt-x` to produce structured outputs with nested items.

- To generate response with more than 2 layers of nested items, seperate them into multipe tasks or utilize nodes.

The following case demonstrates to returning a `Main` class that contains a nested `Sub` class.

**[NOTES]**

- Using `callback` functions to format the final response. (You can try other functions suitable for your use case.)

- Passing parameter: `sub` to the callback function via the `callback_kwargs` variable.

- By default, the outputs of `main_task` are automatically passed to the callback function; you do NOT need to explicitly define them.

- Callback results will be stored in the `callback_output` field of the `TaskOutput` class.


```python
import versionhq as vhq
from pydantic import BaseModel
from typing import Any

# 1. Defines a sub task
class Sub(BaseModel):
    sub1: list[dict[str, Any]]
    sub2: dict[str, Any]

sub_task = vhq.Task(
    description="generate random values that strictly follows the given format.",
    pydantic_output=Sub
)
sub_res = sub_task.execute()

# 2. Defines a main task with callbacks
class Main(BaseModel):
    main1: list[Any] # <= assume expecting to store Sub object.
    main2: dict[str, Any]

def format_response(sub, main1, main2) -> Main:
    if main1:
        main1.append(sub)
    main = Main(main1=main1, main2=main2)
    return main

# 3. Executes
main_task = vhq.Task(
    description="generate random values that strictly follows the given format.",
    pydantic_output=Main,
    callback=format_response,
    callback_kwargs=dict(sub=sub_res.json_dict),
)
res = main_task.execute(context=sub_res.raw) # [Optional] Adding sub_task's response as context.

assert res.callback_output.main1 is not None
assert res.callback_output.main2 is not None
```

To automate these manual setups, refer to <a href="/core/agent-network">AgentNetwork</a> class.
