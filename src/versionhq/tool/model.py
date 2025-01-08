from abc import ABC, abstractmethod
from inspect import signature
from typing import Any, Dict, Callable, Type, Optional, get_args, get_origin
from typing_extensions import Self
from pydantic import InstanceOf, BaseModel, ConfigDict, Field, field_validator, model_validator

from versionhq._utils.cache_handler import CacheHandler


class BaseTool(ABC, BaseModel):
    """
    Abstract class for Tool class.
    """

    class _ArgsSchemaPlaceholder(BaseModel):
        pass

    args_schema: Type[BaseModel] = Field(default_factory=_ArgsSchemaPlaceholder)


    @field_validator("args_schema", mode="before")
    @classmethod
    def _default_args_schema(cls, v: Type[BaseModel]) -> Type[BaseModel]:
        if not isinstance(v, cls._ArgsSchemaPlaceholder):
            return v

        return type(
            f"{cls.__name__}Schema",
            (BaseModel,),
            { "__annotations__": { k: v for k, v in cls._run.__annotations__.items() if k != "return" }},
        )


    @abstractmethod
    def _run(self, *args: Any, **kwargs: Any,) -> Any:
        """any handling"""



class Tool(BaseTool):
    name: str = Field(default=None)
    goal: str = Field(default=None)
    function: Callable = Field(default=None)
    tool_handler: Optional[Dict[str, Any] | Any] = Field(default=None, description="store tool_handler to record the usage of this tool")
    should_cache: bool = Field(default=True, description="whether the tool usage should be cached")
    cache_function: Callable = lambda _args=None, _result=None: True
    cache_handler: Optional[InstanceOf[CacheHandler]] = Field(default=None)


    @model_validator(mode="after")
    def set_up_tool_handler(self) -> Self:
        from versionhq.tool.tool_handler import ToolHandler

        if self.tool_handler and not isinstance(self.tool_handler, ToolHandler):
            ToolHandler(**self.tool_handler)

        else:
            self.tool_handler = ToolHandler(cache_handler=self.cache_handler, should_cache=self.should_cache)

        return self


    @model_validator(mode="after")
    def set_up_function(self) -> Self:
        if self.function is None:
            self.function = self._run
            self._set_args_schema_from_func()
        return self


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
                raise ValueError(f"Failed to parse arguments as JSON: {e}")

        try:
            validated_args = self.args_schema.model_validate(raw_args)
            return validated_args.model_dump()

        except Exception as e:
            raise ValueError(f"Arguments validation failed: {e}")


    def _set_args_schema_from_func(self):
        class_name = f"{self.__class__.__name__}Schema"
        self.args_schema = type(
            class_name,
            (BaseModel,),
            { "__annotations__": {
                k: v for k, v in self._run.__annotations__.items() if k != "return"
            } },
        )
        return self


    def _run(self, *args: Any, **kwargs: Any) -> Any:
        return self.run(*args, **kwargs)


    def run(self, *args, **kwargs) -> Any:
        """
        Use tool and record its usage if should_cache is True.
        """
        from versionhq.tool.tool_handler import ToolHandler

        result = None
        tool_set = ToolSet(tool=self, kwargs={})

        if self.function:
            result = self.function(*args, **kwargs)

        else:
            acceptable_args = self.args_schema.model_json_schema()["properties"].keys()
            acceptable_kwargs = { k: v for k, v in kwargs.items() if k in acceptable_args }
            tool_set = ToolSet(tool=self, kwargs=acceptable_kwargs)

            if self.tool_handler:
                if self.tool_handler.has_called_before(tool_set):
                    self.tool_handler.error = "Agent execution error"

                elif self.tool_handler.cache:
                    result = self.tools_handler.cache.read(tool=tool_set.tool.name, input=tool_set.kwargs)
                    if result is None:
                        parsed_kwargs = self._parse_args(raw_args=acceptable_kwargs)
                        result = self.function(**parsed_kwargs) if self.function else None

            else:
                tool_handler = ToolHandler(last_used_tool=tool_set, cache_handler=self.cache_handler, should_cache=self.should_cache)
                self.tool_handler = tool_handler
                parsed_kwargs = self._parse_args(raw_args=acceptable_kwargs)
                result = self.function(**parsed_kwargs) if self.function else None


        if self.should_cache is True:
            self.tool_handler.record_last_tool_used(tool_set, result, self.should_cache)

        return result


    @property
    def description(self) -> str:
        args_schema = {
            name: {
                "description": field.description,
                "type": self._get_arg_annotations(field.annotation),
            }
            for name, field in self.args_schema.model_fields.items()
        }

        return f"Tool Name: {self.name}\nTool Arguments: {args_schema}\nGoal: {self.goal}"


class ToolSet(BaseModel):
    """
    Store the tool called and any kwargs used.
    """
    tool: InstanceOf[Tool] | Any = Field(..., description="store the tool instance to be called.")
    kwargs: Optional[Dict[str, Any]] = Field(..., description="kwargs passed to the tool")


class InstructorToolSet(BaseModel):
    tool: InstanceOf[Tool] | Any = Field(..., description="store the tool instance to be called.")
    kwargs: Optional[Dict[str, Any]] = Field(..., description="kwargs passed to the tool")


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
