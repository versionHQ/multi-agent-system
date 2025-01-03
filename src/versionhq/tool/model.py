from abc import ABC
from inspect import signature
from typing import Any, Dict, Callable, Type, Optional, get_args, get_origin
from pydantic import InstanceOf, BaseModel, ConfigDict, Field, create_model, field_validator, model_validator

from versionhq._utils.cache_handler import CacheHandler


class Tool(ABC, BaseModel):
    """
    The function that will be executed when the tool is called.
    """

    class ArgsSchema(BaseModel):
        pass

    model_config = ConfigDict()
    name: str = Field(default=None)
    func: Callable = Field(default=None)
    cache_function: Callable = lambda _args=None, _result=None: True
    args_schema: Type[BaseModel] = Field(default_factory=ArgsSchema)
    tool_handler: Optional[Dict[str, Any]] = Field(
        default=None,
        description="store tool_handler to record the usage of this tool. to avoid circular import, set as Dict format",
    )

    @property
    def description(self):
        args_schema = {
            name: { "description": field.description,  "type": Tool._get_arg_annotations(field.annotation) }
            for name, field in self.args_schema.model_fields.items()
        }
        return f"Tool Name: {self.name}\nTool Arguments: {args_schema}\nTool Description: {self.description}"


    @field_validator("args_schema", mode="before")
    @classmethod
    def _default_args_schema(cls, v: Type[BaseModel]) -> Type[BaseModel]:
        if not isinstance(v, cls.ArgsSchema):
            return v

        return type(
            f"{cls.__name__}Schema",
            (BaseModel,),
            {
                "__annotations__": {
                    k: v for k, v in cls._run.__annotations__.items() if k != "return"
                },
            },
        )

    @model_validator(mode="after")
    def set_up_tool_handler_instance(self):
        from versionhq.tool.tool_handler import ToolHandler

        if self.tool_handler:
            ToolHandler(**self.tool_handler)

        return self

    @staticmethod
    def _get_arg_annotations(annotation: type[Any] | None) -> str:
        if annotation is None:
            return "None"

        origin = get_origin(annotation)
        args = get_args(annotation)

        if origin is None:
            return (
                annotation.__name__
                if hasattr(annotation, "__name__")
                else str(annotation)
            )

        if args:
            args_str = ", ".join(Tool._get_arg_annotations(arg) for arg in args)
            return f"{origin.__name__}[{args_str}]"

        return origin.__name__

    def _set_args_schema(self):
        if self.args_schema is None:
            class_name = f"{self.__class__.__name__}Schema"

            self.args_schema = type(
                class_name,
                (BaseModel,),
                {
                    "__annotations__": {
                        k: v
                        for k, v in self._run.__annotations__.items()
                        if k != "return"
                    },
                },
            )

    @classmethod
    def from_composio(
        cls, func: Callable = None, tool_name: str = "Composio tool"
    ) -> "Tool":
        """
        Create a Pydantic BaseModel instance from Composio tools, ensuring the Tool instance has a func to be executed.
        Refer to the `args_schema` from the func signature if any. Else, create an `args_schema`.
        """

        if not func:
            raise ValueError("Params must have a callable 'func' attribute.")

        # args_schema = getattr(tool, "args_schema", None)
        args_fields = {}
        func_signature = signature(func)
        annotations = func_signature.parameters
        for name, param in annotations.items():
            if name != "self":
                param_annotation = (
                    param.annotation if param.annotation != param.empty else Any
                )
                field_info = Field(default=..., description="")
                args_fields[name] = (param_annotation, field_info)

        args_schema = (
            create_model(f"{tool_name}Input", **args_fields)
            if args_fields
            else create_model(f"{tool_name}Input", __base__=BaseModel)
        )

        return cls(name=tool_name, func=func, args_schema=args_schema)


    def run(self, *args, **kwargs) -> Any:
        """
        Use the tool.
        When the tool has a func, execute the func and return any response from the func.
        Else,

        The response will be cascaded to the Task and stored in the TaskOutput.
        """

        result = None

        if self.func is not None:
            result = self.func(
                *args, **kwargs
            )  #! REFINEME - format - json dict, pydantic, raw

        else:
            acceptable_args = self.args_schema.model_json_schema()["properties"].keys()
            arguments = {k: v for k, v in kwargs.items() if k in acceptable_args}
            tool_called = ToolCalled(tool=self, arguments=arguments)

            if self.tool_handler:
                if self.tool_handler.has_called_before(tool_called):
                    self.tool_handler.error = "Agent execution error."

                elif self.tool_handler.cache:
                    result = self.tools_handler.cache.read(
                        tool=tool_called.tool.name, input=tool_called.arguments
                    )
                    # from_cache = result is not None
                    if result is None:
                        result = self.invoke(input=arguments)

            else:
                from versionhq.tool.tool_handler import ToolHandler

                tool_handler = ToolHandler(
                    last_used_tool=tool_called, cache=CacheHandler()
                )
                self.tool_handler = tool_handler
                result = self.invoke(input=arguments)

        return result


class ToolCalled(BaseModel):
    """
    Store the tool called and any kwargs used.
    """

    tool: InstanceOf[Tool] = Field(..., description="store the tool instance to be called.")
    arguments: Optional[Dict[str, Any]] = Field(..., description="kwargs passed to the tool")


class InstructorToolCalled(BaseModel):
    tool: InstanceOf[Tool] = Field(..., description="store the tool instance to be called.")
    arguments: Optional[Dict[str, Any]] = Field(..., description="kwargs passed to the tool")


class CacheTool(BaseModel):
    """
    Default tools to hit the cache.
    """

    name: str = "Hit Cache"
    cache_handler: CacheHandler = Field(default_factory=CacheHandler)

    def hit_cache(self, key):
        split = key.split("tool:")
        tool = split[1].split("|input:")[0].strip()
        tool_input = split[1].split("|input:")[1].strip()
        return self.cache_handler.read(tool, tool_input)
