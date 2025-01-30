import logging
import json
import os
import sys
import threading
import warnings
import litellm
from litellm import JSONSchemaValidationError
from abc import ABC
from dotenv import load_dotenv
from litellm import get_supported_openai_params
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Type
from typing_extensions import Self

from pydantic import UUID4, BaseModel, Field, PrivateAttr, field_validator, model_validator, create_model, InstanceOf, ConfigDict
from pydantic_core import PydanticCustomError

from openai import OpenAI

from versionhq.llm.llm_vars import LLM_CONTEXT_WINDOW_SIZES, LLM_API_KEY_NAMES, LLM_BASE_URL_KEY_NAMES, MODELS, PARAMS, SchemaType
from versionhq.task import TaskOutputFormat
from versionhq.task.model import ResponseField, Task
from versionhq.tool.model import Tool, ToolSet
from versionhq._utils.logger import Logger


load_dotenv(override=True)
LITELLM_API_KEY = os.environ.get("LITELLM_API_KEY")
LITELLM_API_BASE = os.environ.get("LITELLM_API_BASE")
DEFAULT_CONTEXT_WINDOW_SIZE = int(8192 * 0.75)
DEFAULT_MODEL_NAME = os.environ.get("DEFAULT_MODEL_NAME")

proxy_openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"), organization="versionhq", base_url=LITELLM_API_BASE)
openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


class FilteredStream:
    def __init__(self, original_stream):
        self._original_stream = original_stream
        self._lock = threading.Lock()

    def write(self, s) -> int:
        with self._lock:
            if (
                "Give Feedback / Get Help: https://github.com/BerriAI/litellm/issues/new"
                in s
                or "LiteLLM.Info: If you need to debug this error, use `os.environ['LITELLM_LOG'] = 'DEBUG'`"
                in s
            ):
                return 0
            return self._original_stream.write(s)

    def flush(self):
        with self._lock:
            return self._original_stream.flush()


@contextmanager
def suppress_warnings():
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = FilteredStream(old_stdout)
        sys.stderr = FilteredStream(old_stderr)

        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


class LLM(BaseModel):
    """
    An LLM class to store params except for response formats which will be given in the task handling process.
    Use LiteLLM to connect with the model of choice.
    Some optional params are passed by the agent, else follow the default settings of the model provider.
    Ref. https://docs.litellm.ai/docs/completion/input
    """

    _logger: Logger = PrivateAttr(default_factory=lambda: Logger(verbose=True))
    _init_model_name: str = PrivateAttr(default=None)
    _tokens: int = PrivateAttr(default=0) # accumulate total tokens used for the call
    model_config = ConfigDict(extra="allow")

    model: str = Field(default=DEFAULT_MODEL_NAME)
    provider: Optional[str] = Field(default=None, description="model provider or custom model provider")
    base_url: Optional[str] = Field(default=None, description="api base of the model provider")
    api_key: Optional[str] = Field(default=None, description="api key of the model provider")

    # optional params
    timeout: Optional[float | int] = Field(default=None)
    max_tokens: Optional[int] = Field(default=None)
    max_completion_tokens: Optional[int] = Field(default=None)
    context_window_size: Optional[int] = Field(default=DEFAULT_CONTEXT_WINDOW_SIZE)
    callbacks: List[Any] = Field(default_factory=list)
    temperature: Optional[float] = Field(default=None)
    top_p: Optional[float] = Field(default=None)
    n: Optional[int] = Field(default=None)
    stop: Optional[str | List[str]] = Field(default=None)
    presence_penalty: Optional[float] = Field(default=None)
    frequency_penalty: Optional[float] = Field(default=None)
    logit_bias: Optional[Dict[int, float]] = Field(default=None)
    seed: Optional[int] = Field(default=None)
    logprobs: Optional[bool] = Field(default=None)
    top_logprobs: Optional[int] = Field(default=None)
    response_format: Optional[Any] = Field(default=None)
    tools: Optional[List[Dict[str, Any]]] = Field(default_factory=list, description="store a list of tool properties")

    # LiteLLM specific fields
    api_base: Optional[str] = Field(default=None, description="litellm specific field - api base of the model provider")
    api_version: Optional[str] = Field(default=None)
    num_retries: Optional[int] = Field(default=2)
    context_window_fallback_dict: Optional[Dict[str, Any]] = Field(default=None, description="A mapping of model to use if call fails due to context window error")
    fallbacks: Optional[List[Any]]= Field(default=None, description="A list of model names + params to be used, in case the initial call fails")
    metadata: Optional[Dict[str, Any]] = Field(default=None)

    litellm.drop_params = True
    litellm.set_verbose = True
    os.environ['LITELLM_LOG'] = 'DEBUG'

    @model_validator(mode="after")
    def validate_base_params(self) -> Self:
        """
        1) Set up a valid model name with the provider name using the MODEL list.
        * Assign a default model and provider based on the given information when no model key is found in the MODEL list.

        2) Set up other base parameters for the model and LiteLLM.
        """

        if self.model is None:
            self._logger.log(level="error", message="Model name is missing.", color="red")
            raise PydanticCustomError("model_missing", "The model name must be provided.", {})


        self._init_model_name = self.model
        self.model = None
        self._tokens = 0

        if self.provider and MODELS.get(self.provider):
            provider_model_list = MODELS.get(self.provider)
            for item in provider_model_list:
                if self.model is None:
                    if item == self._init_model_name:
                        self.model = item
                    elif self._init_model_name in item and self.model is None:
                        self.model = item
                    else:
                        temp_model = provider_model_list[0]
                        self._logger.log(level="warning", message=f"The provided model: {self._init_model_name} is not in the list. We'll assign a model: {temp_model} from the selected model provider: {self.provider}.", color="yellow")
                        self.model = temp_model

        else:
            for k, v in MODELS.items():
                for item in v:
                    if self.model is None:
                        if self._init_model_name == item:
                            self.model = item
                            self.provider = k

                        elif self.model is None and self._init_model_name in item:
                            self.model = item
                            self.provider = k

            if self.model is None:
                self._logger.log(level="warning", message=f"The provided model \'{self.model}\' is not in the list. We'll assign a default model.", color="yellow")
                self.model = DEFAULT_MODEL_NAME
                self.provider = "openai"


        if self.callbacks:
            self._set_callbacks(self.callbacks)

        self.context_window_size = self._get_context_window_size()

        api_key_name = self.provider.upper() + "_API_KEY" if self.provider else None
        if api_key_name:
            self.api_key = os.environ.get(api_key_name, None)

        base_url_key_name = self.provider.upper() + "_API_BASE" if self.provider else None
        if base_url_key_name:
            self.base_url = os.environ.get(base_url_key_name)
            self.api_base = self.base_url

        return self


    def _create_valid_params(self, config: Dict[str, Any], provider: str = None) -> Dict[str, Any]:
        params = dict()
        valid_keys = list()

        if not provider:
            valid_keys = PARAMS.get("litellm") + PARAMS.get("common") + PARAMS.get(self.provider) if self.provider else PARAMS.get("litellm") + PARAMS.get("common")
        else:
            valid_keys = PARAMS.get("common") + PARAMS.get(self.provider)

        for item in valid_keys:
            if hasattr(self, item) and getattr(self, item):
                params[item] = getattr(self, item)
            elif item in config:
                params[item] = config[item]

        return params


    def call(
        self,
        messages: List[Dict[str, str]],
        response_format: Optional[Dict[str, Any]] = None,
        tools: Optional[List[Tool | ToolSet | Any ]] = None,
        config: Optional[Dict[str, Any]] = {}, # any other conditions to pass on to the model.
        tool_res_as_final: bool = False
    ) -> str:
        """
        Execute LLM based on the agent's params and model params.
        """

        litellm.drop_params = True

        with suppress_warnings():
            if len(self.callbacks) > 0:
                self._set_callbacks(self.callbacks) # passed by agent

            try:
                provider = self.provider if self.provider else "openai"
                self.response_format = { "type": "json_object" } if tool_res_as_final == True else response_format

                if not tools:
                    params = self._create_valid_params(config=config)
                    res = litellm.completion(messages=messages, stream=False, **params)
                    self._tokens += int(res["usage"]["total_tokens"])
                    return res["choices"][0]["message"]["content"]

                else:
                    self.tools = [item.tool.properties if isinstance(item, ToolSet) else item.properties for item in tools]

                    if provider == "openai":
                        params = self._create_valid_params(config=config, provider=provider)
                        res = openai_client.chat.completions.create(messages=messages, model=self.model, tools=self.tools)
                        tool_calls = res.choices[0].message.tool_calls
                        tool_res = ""

                        for item in tool_calls:
                            func_name = item.function.name
                            func_args = item.function.arguments

                            if not isinstance(func_args, dict):
                                try:
                                    func_args = json.loads(json.dumps(eval(str(func_args))))
                                except:
                                    pass

                            for tool in tools:
                                if isinstance(tool, ToolSet) and (tool.tool.name == func_name or tool.tool.func.__name__ == func_name or func_name == "random_func"):
                                    tool_instance = tool.tool
                                    args = tool.kwargs
                                    tool_res_to_add = tool_instance.run(params=args)

                                    if tool_res_as_final:
                                        tool_res += str(tool_res_to_add)
                                    else:
                                        messages.append(res.choices[0].message)
                                        messages.append({ "role": "tool", "tool_call_id": item.id, "content": str(tool_res_to_add) })

                                else:
                                    try:
                                        tool_res_to_add = tool.run(params=func_args)
                                        if tool_res_as_final:
                                            tool_res += str(tool_res_to_add)
                                        else:
                                            messages.append(res.choices[0].message)
                                            messages.append({ "role": "tool", "tool_call_id": item.id, "content": str(tool_res_to_add) })
                                    except:
                                        pass

                        if tool_res_as_final:
                            return tool_res
                        else:
                            res = openai_client.chat.completions.create(messages=messages, model=self.model, tools=self.tools)
                            self._tokens += int(res["usage"]["total_tokens"])
                            return res.choices[0].message.content

            except JSONSchemaValidationError as e:
                self._logger.log(level="error", message="Raw Response: {}".format(e.raw_response), color="red")
                raise e

            except Exception as e:
                self._logger.log(level="error", message=f"{self.model} failed to execute: {str(e)}", color="red")
                if "litellm.RateLimitError" in str(e):
                    raise e


    def _supports_function_calling(self) -> bool:
        try:
            params = get_supported_openai_params(model=self.model)
            return "response_format" in params
        except Exception as e:
            self._logger.log(level="error", message=f"Failed to get supported params: {str(e)}", color="red")
            return False


    def _supports_stop_words(self) -> bool:
        try:
            params = get_supported_openai_params(model=self.model)
            return "stop" in params
        except Exception as e:
            self._logger.log(level="error", message=f"Failed to get supported params: {str(e)}", color="red")
            return False


    def _get_context_window_size(self) -> int:
        """
        Only use 75% of the context window size to avoid cutting the message in the middle.
        """
        return int(LLM_CONTEXT_WINDOW_SIZES.get(self.model) * 0.75) if LLM_CONTEXT_WINDOW_SIZES.get(self.model) is not None else DEFAULT_CONTEXT_WINDOW_SIZE


    def _set_callbacks(self, callbacks: List[Any]):
        callback_types = [type(callback) for callback in callbacks]
        for callback in litellm.success_callback[:]:
            if type(callback) in callback_types:
                litellm.success_callback.remove(callback)

        for callback in litellm._async_success_callback[:]:
            if type(callback) in callback_types:
                litellm._async_success_callback.remove(callback)

        litellm.callbacks = callbacks
