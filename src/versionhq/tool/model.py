from abc import ABC, abstractmethod
from inspect import signature
from typing import Any, Dict, Callable, Type, Optional, get_args, get_origin, get_type_hints
from typing_extensions import Self
from pydantic import InstanceOf, BaseModel, ConfigDict, Field, field_validator, model_validator, PrivateAttr, create_model
from pydantic_core import PydanticCustomError

from versionhq.llm.llm_vars import SchemaType
from versionhq.tool.cache_handler import CacheHandler
from versionhq._utils.logger import Logger


class BaseTool(ABC, BaseModel):
    """
    Abstract class for Tool class.
    """

    class ArgsSchemaPlaceholder(BaseModel):
        pass

    _logger: Logger = PrivateAttr(default_factory=lambda: Logger(verbose=True))

    object_type: str = Field(default="function")
    name: str = Field(default=None)
    description: str = Field(default=None)
    properties: Dict[str, Any] = Field(default_factory=dict, description="for llm func calling")
    args_schema: Type[BaseModel] = Field(default_factory=ArgsSchemaPlaceholder)

    tool_handler: Optional[Dict[str, Any] | Any] = Field(default=None, description="store tool_handler to record the tool usage")
    should_cache: bool = Field(default=True, description="whether the tool usage should be cached")
    cache_function: Callable = lambda _args=None, _result=None: True
    cache_handler: Optional[InstanceOf[CacheHandler]] = Field(default=None)


    @field_validator("args_schema", mode="before")
    @classmethod
    def _default_args_schema(cls, v: Type[BaseModel]) -> Type[BaseModel]:
        if not isinstance(v, cls.ArgsSchemaPlaceholder):
            return v

        return type(
            f"{cls.__name__}Schema",
            (BaseModel,),
            { "__annotations__": { k: v for k, v in cls._run.__annotations__.items() if k != "return" }},
        )


    @field_validator("properties", mode="before")
    @classmethod
    def _default_properties(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        p, r = dict(), list()
        for k, v in cls._run.__annotations__.items():
            if k != "return":
                p.update({ k: { "type": SchemaType(type(v)).convert(), "name": k, }} )
                r.append(k)

        return {
            "type": cls.object_type,
            "function": {
                "name": cls.name.replace(" ", "_"),
                "description": cls.description if cls.description else "",
                "parameters": {
                    "type": "object",
                    "properties": p,
                    "required": r,
                    "additionalProperties": False
                },
                "strict": True
            }
        }

    @model_validator(mode="after")
    def set_up_tool_handler(self) -> Self:
        from versionhq.tool.tool_handler import ToolHandler

        if self.tool_handler and not isinstance(self.tool_handler, ToolHandler):
            ToolHandler(**self.tool_handler)

        else:
            self.tool_handler = ToolHandler(cache_handler=self.cache_handler, should_cache=self.should_cache)

        return self

    @abstractmethod
    def _run(self, *args: Any, **kwargs: Any,) -> Any:
        """any handling"""


    @staticmethod
    def _get_arg_annotations(annotation: type[Any] | None) -> str:
        if annotation is None:
            return "None"

        origin = get_origin(annotation)
        args = get_args(annotation)

        if origin is None:
            return annotation.__name__ if hasattr(annotation, "__name__") else str(annotation)

        if args:
            args_str = ", ".join(Tool._get_arg_annotations(arg) for arg in args)
            return f"{origin.__name__}[{args_str}]"

        return origin.__name__


    def _parse_args(self, raw_args: str | dict) -> dict:
        """
        Parse and validate the input arguments against the schema
        """
        if isinstance(raw_args, str):
            try:
                import json
                raw_args = json.loads(raw_args)
            except json.JSONDecodeError as e:
                raise ValueError(f"Failed to parse arguments as JSON: {str(e)}")

        try:
            validated_args = self.args_schema.model_validate(raw_args)
            return validated_args.model_dump()

        except Exception as e:
            raise ValueError(f"Arguments validation failed: {str(e)}")


    def _create_schema(self) -> type[BaseModel]:
        """
        Create a Pydantic schema from a function's signature
        """
        import inspect

        sig = inspect.signature(self.func)
        type_hints = get_type_hints(self.func)
        fields = {}
        for param_name, param in sig.parameters.items():
            if param_name in ("self", "cls"):
                continue

            annotation = type_hints.get(param_name, Any)
            default = ... if param.default == param.empty else param.default
            fields[param_name] = (annotation, Field(default=default))

        schema_name = f"{self.func.__name__.title()}Schema"
        return create_model(schema_name, **fields)



class Tool(BaseTool):
    func: Callable = Field(default=None)


    @model_validator(mode="after")
    def validate_func(self) -> Self:
        if not self.func and not self._run:
            self._logger.log(level="error", message=f"Tool must have a function", color="red")
            raise PydanticCustomError("function_missing", f"Function is missing in the tool.", {})

        elif self.func and not isinstance(self.func, Callable):
            self._logger.log(level="error", message=f"The tool is missing a valid function", color="red")
            raise PydanticCustomError("invalid_function", f"The value in the function field must be callable.", {})

        else:
            try:
                self.args_schema = self._create_schema_from_function()
                self._validate_function_signature()

            except Exception as e:
                self._logger.log(level="error", message=f"The tool is missing a valid function: {str(e)}", color="red")
                raise PydanticCustomError("invalid_function", f"Invalid function: {str(e)}", {})

        return self


    @model_validator(mode="after")
    def set_up_name(self) -> Self:
        if not self.name:
            self.name = self.func.__name__ if self.func.__name__ != "<lambda>" else "random_func"

        return self


    @model_validator(mode="after")
    def set_up_description(self) -> Self:
        if not self.description:
            if not self.args_schema:
                self.args_schema = self._default_args_schema(self)

            args_schema = {
                name: {
                    "description": field.description if field.description else "",
                    "type": self._get_arg_annotations(field.annotation),
                }
                for name, field in self.args_schema.model_fields.items()
            }
            self.description = f"Tool: {self.name}\nArgs: {args_schema}"

        return self


    @model_validator(mode="after")
    def set_up_args_schema(self) -> Self:
        """
        Set up args schema based on the given function.
        """
        if self.func:
            self.args_schema = self._create_schema_from_function()
        return self


    @model_validator(mode="after")
    def set_up_func_calling_properties(self) -> Self:
        """
        Format function_calling params from args_schema.
        """

        p, r = dict(), list()
        if not self.args_schema:
            self.args_schema = self.set_up_args_schema()

        for name, field in self.args_schema.model_fields.items():
            if name != "kwargs" and name != "args":
                p.update(
                    {
                        name: {
                            "description": field.description if field.description else "",
                            "type": SchemaType(self._get_arg_annotations(field.annotation)).convert(),
                        }
                    }
                )
                r.append(name)

        properties = {
            "type": self.object_type,
            "function": {
                "name": self.name.replace(" ", "_"),
                "description": self.description if self.description else "a tool function to execute",
                "parameters": {
                    "type": "object",
                    "properties": p,
                    "required": r,
                    "additionalProperties": False
                },
                "strict": True,
            },
        }
        self.properties = properties
        return self


    def _create_schema_from_function(self) -> type[BaseModel]:
        """
        Create a Pydantic schema from a function's signature
        """
        import inspect

        sig = inspect.signature(self.func)
        type_hints = get_type_hints(self.func)
        fields = {}
        for param_name, param in sig.parameters.items():
            if param_name in ("self", "cls"):
                continue

            annotation = type_hints.get(param_name, Any)
            default = ... if param.default == param.empty else param.default
            fields[param_name] = (annotation, Field(default=default))

        schema_name = f"{self.func.__name__.title()}Schema"
        return create_model(schema_name, **fields)


    def _validate_function_signature(self) -> None:
        """
        Validate that the function signature matches the args schema.
        """

        import inspect

        sig = inspect.signature(self.func)
        schema_fields = self.args_schema.model_fields

        for param_name, param in sig.parameters.items():
            if param_name in ("self", "cls"):
                continue

            if param.kind in (inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL):
                continue

            if param.default == inspect.Parameter.empty:
                if param_name not in schema_fields:
                    raise ValueError(f"Required function parameter '{param_name}' not found in args_schema")


    def _run(self, *args: Any, **kwargs: Any) -> Any:
        return self.func(*args, **kwargs)


    def _handle_toolset(self, params: Dict[str, Any] = None) -> Any:
        """
        Return cached results or run the function and record the results.
        """

        from versionhq.tool.tool_handler import ToolHandler

        if not self.args_schema:
            self.args_schema = self._create_schema_from_function()

        result = None
        acceptable_args = self.args_schema.model_json_schema()["properties"].keys()
        acceptable_kwargs = { k: v for k, v in params.items() if k in acceptable_args } if params else dict()
        parsed_kwargs = self._parse_args(raw_args=acceptable_kwargs)
        tool_set = ToolSet(tool=self, kwargs=acceptable_kwargs)

        if not self.tool_handler or not isinstance(self.tool_handler, ToolHandler):
            self.tool_handler = ToolHandler(last_used_tool=tool_set, cache_handler=self.cache_handler, should_cache=self.should_cache)

        try:
            if self.tool_handler.has_called_before(tool_set) or self.tool_handler.cache:
                result = self.tool_handler.cache.read(tool_name=tool_set.tool.name, input=str(tool_set.kwargs))

            if not result:
                result = self.func(**parsed_kwargs)

            if self.should_cache is True:
                self.tool_handler.record_last_tool_used(last_used_tool=tool_set, output=result, should_cache=self.should_cache)

            return result

        except:
            self.tool_handler.error = "Agent error"
            return result


    def run(self, params: Dict[str, Any] = None) -> Any:
        """
        Execute a tool using a toolset and cached tools
        """
        result = self._handle_toolset(params)
        return result


class ToolSet(BaseModel):
    """
    Store the tool called and any kwargs used. (The tool name and kwargs will be stored in the cache.)
    """
    tool: InstanceOf[Tool]| Type[Tool] = Field(..., description="store the tool instance to be called.")
    kwargs: Optional[Dict[str, Any]] = Field(..., description="kwargs passed to the tool")
