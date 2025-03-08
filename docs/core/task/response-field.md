# Response Field

<class>`class` versionhq.task.model.<bold>ResponseField<bold></class>

A Pydantic class to store response formats to generate a structured response in JSON.

**Quick Start**

Define a response format with field titles and data types.


```python
import versionhq as vhq

response_field = vhq.ResponseField(
  title="summary",
  data_type=str,
  nullable=False, # default = False. Explicitly mentioned.
)

## Agent output:
#  "summary": <AGENT_RESPONSE_IN_STRING>
```

### Object

`[var]`<bold>`properties: List[InstanceOf[ResponseField]] = None`</bold>

To format an object, add `ResponseField` objects to the `properties` fields.

Missing properties for dict will trigger an error.

```python
import versionhq as vhq

response_field = vhq.ResponseField(
  title="dict-summary",
  data_type=dict,
  nullable=False,
  properties=[
    vhq.ResponseField(title="summary-1", data_type=str),
    vhq.ResponseField(title="summary-2", data_type=int),
  ]
)

## Agent output:
#  dict-summary: {
#   "summary-1": <AGENT_RESPONSE_1_IN_STRING>,
#   "summary-2": <AGENT_RESPONSE_2_IN_INTEGER>,
# }
```

<hr />

### List

`[var]`<bold>`items: Optional[Type] = None`</bold>

To format a list, add data types of the list items to the `items` field.

Missing items for list will trigger an error.

```python
import versionhq as vhq

response_field = vhq.ResponseField(
  title="list-summary",
  data_type=list,
  nullable=False,
  items=str
)

## Agent output:
#  list-summary: [
#     <AGENT_RESPONSE_1_IN_STRING>,
#     <AGENT_RESPONSE_2_IN_STRING>,
#     ...
# ]
```

<hr />

## Nesting

Agents can handle **one layer** of nested items usign `properties` and `items` fields.

We highly recommend to use `gemini-x` or `gpt-x` to get stable results.

### Object in List

```python
import versionhq as vhq

list_with_objects = vhq.ResponseField(
  title="parent",
  data_type=list,
  items=dict,
  properties=[
    vhq.ResponseField(title="nest-1", data_type=str),
    vhq.ResponseField(title="nest-2", data_type=float),
  ]
)

# Agent output
# parent: [
#   { "nest-1": <AGENT_RESOPONSE_IN_STRING>},
#   { "nest-2": <AGENT_RESOPONSE_IN_NUMBER>},
# ]
```

<hr />

### List in List

```python
import versionhq as vhq

list_with_list = vhq.ResponseField(
  title="parent",
  data_type=list,
  items=list
)

# Agent output
# parent: [
#   [<AGENT_RESOPONSE_IN_STRING>, ...],
#   [<AGENT_RESOPONSE_IN_STRING>, ...]
#   ...
# ]
```

<hr />

### List in Object

```python
import versionhq as vhq

dict_with_list = vhq.ResponseField(
  title="parent",
  data_type=dict,
  properties=[
    vhq.ResponseField(title="nest-1", data_type=list, items=str),
    vhq.ResponseField(title="nest-2", data_type=list, items=int),
  ]
)

# Agent output
# parent: {
#   nest-1: [<AGENT_RESOPONSE_IN_STRING>, ...],
#   nest-2: [<AGENT_RESOPONSE_IN_INTEGER>, ...]
# }
```

<hr />

### Object in Object

```python
import versionhq as vhq

dict_with_dict = vhq.ResponseField(
  title="parent",
  data_type=dict,
  properties=[
    vhq.ResponseField(title="nest-1", data_type=dict, properties=[
      vhq.ResponseField(title="nest-1-1", data_type=str)
    ]),
    vhq.ResponseField(title="nest-2", data_type=list, items=int),
  ]
)

# Agent output
# parent: {
#   nest-1: { nest-1-1: <AGENT_RESOPONSE_IN_STRING>, },
#   nest-2: [<AGENT_RESOPONSE_IN_INTEGER>, ...]
# }
```

### Config

`[var]`<bold>`config: Optional[Dict[str, Any]] = {}`</bold>

You can add other configs you want to pass to the LLM.

```python
import versionhq as vhq

response_field = vhq.ResponseField(
  title="summary-with-config",
  data_type=str,
  nullable=False,
  config=dict(required=False, )
)

# Agent output:
# summary-with-config: <AGENT_RESPONSE_IN_STRING>
```

Ref. <a href="/core/task/reference/#responsefield">List of variables</a>
