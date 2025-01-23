from typing import Callable
from pydantic import BaseModel

from versionhq.tool.model import Tool


def tool(*args):
    """
    Decorator to create a tool from a function.
    """

    def create_tool(tool_name: str) -> Callable:
        def _make_tool(func: Callable) -> Tool:
            if func.__doc__ is None:
                raise ValueError("Function must have a docstring")

            if func.__annotations__ is None:
                raise ValueError("Function must have type annotations")

            class_name = "".join(tool_name.split()).title()
            args_schema = type(
                class_name,
                (BaseModel,),
                {
                    "__annotations__": {
                        k: v for k, v in func.__annotations__.items() if k != "return"
                    },
                },
            )
            return Tool(name=tool_name, func=func, args_schema=args_schema)

        return _make_tool

    if len(args) == 1 and callable(args[0]):
        return create_tool(args[0].__name__)(args[0])

    elif len(args) == 1 and isinstance(args[0], str):
        return create_tool(args[0])

    else:
        raise ValueError("Invalid arguments")
