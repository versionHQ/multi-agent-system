from typing import Dict, Optional, Type, List, Any, TypeVar

from pydantic import BaseModel, Field, InstanceOf

from versionhq.llm.llm_vars import SchemaType
from versionhq.llm.model import LLM


"""
Structure a response schema (json schema) from the given Pydantic model.
"""


class StructuredObject:
    """
    A class to store the structured dictionary.
    """
    provider: str = "openai"
    field: Type[Field]

    title: str
    dtype: str = "object"
    properties: Dict[str, Dict[str, str]] = dict()
    required: List[str] = list()
    additionalProperties: bool = False

    def __init__(self, name, field: Type[Field], provider: str | InstanceOf[LLM] = "openai"):
        self.title = name
        self.field = field
        self.dtype = "object"
        self.additionalProperties = False
        self.provider = provider if isinstance(provider, str) else provider.provider

    def _format(self):
        if not self.field:
            pass
        else:
            description = self.field.description if hasattr(self.field, "description") and self.field.description is not None else ""
            self.properties.update({"item": { "type": SchemaType(self.field.annotation.__args__).convert() }})
            self.required.append("item")

            return {
                self.title: {
                    "type": self.dtype,
                    "description": description,
                    "properties": self.properties,
                    "additionalProperties": self.additionalProperties,
                    "required": self.required
                }
            }



class StructuredList:
    """
    A class to store a structured list with 1 nested object.
    """
    provider: str = "openai"
    field: Type[Field]
    title: str = ""
    dtype: str = "array"
    items: Dict[str, Dict[str, str]] = dict()

    def __init__(self, name, field: Type[Field], provider: str | LLM = "openai"):
        self.provider = provider if isinstance(provider, str) else provider.provider
        self.field = field
        self.title = name
        self.dtype = "array"
        self.items = dict()


    def _format(self):
        field = self.field
        if not field:
            pass
        else:
            description = "" if field.description is None else field.description
            props = {}

            for item in field.annotation.__args__:
                nested_object_type = item.__origin__ if hasattr(item, "__origin__") else item

                if nested_object_type == dict:
                    props.update({
                        "nest":  {
                            "type": "object",
                            "properties": { "item": { "type": "string"} }, #! REFINEME - field title <>`item`
                            "required": ["item",],
                            "additionalProperties": False
                        }})

                elif nested_object_type == list:
                    props.update({
                        # "nest":  {
                            "type": "array",
                            "items": { "type": "string" } , #! REFINEME - field title <>`item`
                        # }
                        })
                else:
                    props.update({ "type": SchemaType(nested_object_type).convert() })

            self.items = { **props }
            return {
                 self.title: {
                    "type": self.dtype,
                    "description": description,
                    "items": self.items,
                }
            }


class StructuredOutput(BaseModel):
    response_format: Any = None
    provider: str = "openai"
    applicable_models: List[InstanceOf[LLM] | str] = list()
    name: str = ""
    schema: Dict[str, Any] = dict(type="object", additionalProperties=False, properties=dict(), required=list())


    def _format(self, **kwargs):
        if self.response_format is None:
            pass

        self.name = self.response_format.__name__

        for name, field in self.response_format.model_fields.items():
            self.schema["required"].append(name)

            if hasattr(field.annotation, "__origin__") and field.annotation.__origin__ == dict:
                self.schema["properties"].update(StructuredObject(name=name, field=field)._format())

            elif hasattr(field.annotation, "__origin__") and field.annotation.__origin__ == list:
                self.schema["properties"].update(StructuredList(name=name, field=field)._format())
            else:
                self.schema["properties"].update({ name: { "type": SchemaType(field.annotation).convert(), **kwargs }})

        return {
            "type": "json_schema",
            "json_schema": { "name": self.name, "schema": self.schema }
        }
