import json
import os
import sys
import threading
import warnings
from dotenv import load_dotenv
from contextlib import contextmanager
from typing import Any, Dict, List, Optional
from typing_extensions import Self

import litellm
from litellm import JSONSchemaValidationError, get_supported_openai_params
from pydantic import BaseModel, Field, PrivateAttr, model_validator, ConfigDict

from versionhq.llm.llm_vars import LLM_CONTEXT_WINDOW_SIZES, MODELS, PARAMS, PROVIDERS, ENDPOINT_PROVIDERS, ENV_VARS
from versionhq.tool.model import Tool, ToolSet
from versionhq._utils import Logger


load_dotenv(override=True)
LITELLM_API_KEY = os.environ.get("LITELLM_API_KEY")
LITELLM_API_BASE = os.environ.get("LITELLM_API_BASE")
DEFAULT_CONTEXT_WINDOW_SIZE = int(8192 * 0.75)
DEFAULT_MODEL_NAME = os.environ.get("DEFAULT_MODEL_NAME", "gpt-4o-mini")
DEFAULT_MODEL_PROVIDER_NAME = os.environ.get("DEFAULT_MODEL_PROVIDER_NAME", "openai")

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
        litellm.set_verbose = False
        warnings.filterwarnings(action="ignore")
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
    An LLM class to store params to send to the LLM. Use LiteLLM or custom providers for the endpoint.
    """

    _logger: Logger = PrivateAttr(default_factory=lambda: Logger(verbose=True))
    _init_model_name: str = PrivateAttr(default=None)
    _tokens: int = PrivateAttr(default=0) # aggregate number of tokens consumed

    model: str = Field(default=None)
    provider: Optional[str] = Field(default=None, description="model provider")
    endpoint_provider: Optional[str] = Field(default=None, description="custom endpoint provider for pass through llm call. require base_url")
    base_url: Optional[str] = Field(default=None, description="api base url for endpoint provider")

    # optional params
    response_format: Optional[Any] = Field(default=None)
    llm_config: Optional[Dict[str, Any]] = Field(default_factory=dict, description="stores valid llm config params")
    callbacks: Optional[List[Any]] = Field(default_factory=list)
    tools: Optional[List[Dict[str, Any]]] = Field(default_factory=list, description="stores a list of tool properties")
    timeout: Optional[float | int] = Field(default=None)
    context_window_size: Optional[int] = Field(default=DEFAULT_CONTEXT_WINDOW_SIZE)

    # LiteLLM specific config
    api_base: Optional[str] = Field(default=None, description="litellm specific field - api base of the model provider")
    api_version: Optional[str] = Field(default=None)
    num_retries: Optional[int] = Field(default=1)
    context_window_fallback_dict: Optional[Dict[str, Any]] = Field(default=None, description="A mapping of model to use if call fails due to context window error")
    fallbacks: Optional[List[Any]]= Field(default=None, description="A list of model names + params to be used, in case the initial call fails")
    metadata: Optional[Dict[str, Any]] = Field(default=None)

    model_config = ConfigDict(extra="allow")

    os.environ['LITELLM_LOG'] = 'DEBUG'


    @model_validator(mode="after")
    def validate_model_and_provider(self) -> Self:
        """
        Validate the given model, provider, interface provider.
        """
        litellm.drop_params = True

        self._init_model_name = self.model

        if self.model is None and self.provider is None:
            self.model = DEFAULT_MODEL_NAME
            self.provider = DEFAULT_MODEL_PROVIDER_NAME

        elif self.model is None and self.provider:
            if self.provider not in PROVIDERS:
                self._logger.log(level="warning", message=f"Invalid model provider is provided. We will assign a default model.", color="yellow")
                self.model = DEFAULT_MODEL_NAME
                self.provider = DEFAULT_MODEL_PROVIDER_NAME

            else:
                provider_model_list = MODELS.get(self.provider)
                if provider_model_list:
                    self.model = provider_model_list[0]
                    self.provider = self.provider
                else:
                    self._logger.log(level="warning", message=f"This provider has not models to be called. We will assign a default model.", color="yellow")
                    self.model = DEFAULT_MODEL_NAME
                    self.provider = DEFAULT_MODEL_PROVIDER_NAME

        elif self.model and self.provider is None:
            model_match = [
                item for item in [
                    [val for val in v if val == self.model][0] for k, v in MODELS.items() if [val for val in v if val == self.model]
                ] if item
            ]
            model_partial_match = [
                item for item in [
                    [val for val in v if val.find(self.model) != -1][0] for k, v in MODELS.items() if [val for val in v if val.find(self.model) != -1]
                ] if item
            ]
            provider_match = [k for k, v in MODELS.items() if k == self.model]

            if model_match:
                self.model = model_match[0]
                self.provider = [k for k, v in MODELS.items() if self.model in v][0]

            elif model_partial_match:
                self.model = model_partial_match[0]
                self.provider = [k for k, v in MODELS.items() if [item for item in v if item.find(self.model) != -1]][0]

            elif provider_match:
                provider = provider_match[0]
                if self.MODELS.get(provider):
                    self.provider = provider
                    self.model = self.MODELS.get(provider)[0]
                else:
                    self.provider = DEFAULT_MODEL_PROVIDER_NAME
                    self.model = DEFAULT_MODEL_NAME

            else:
                self.model = DEFAULT_MODEL_NAME
                self.provider = DEFAULT_MODEL_PROVIDER_NAME

        else:
            provider_model_list = MODELS.get(self.provider)
            if self.model not in provider_model_list:
                self._logger.log(level="warning", message=f"The provided model: {self._init_model_name} is not in the list. We will assign a default model.", color="yellow")
                self.model = DEFAULT_MODEL_NAME
                self.provider = DEFAULT_MODEL_PROVIDER_NAME

        # trigger pass-through custom endpoint.
        if self.provider in ENDPOINT_PROVIDERS:
            self.endpoint_provider = self.provider

        return self


    @model_validator(mode="after")
    def setup_config(self) -> Self:
        """
        Set up valid config params after setting up a valid model, provider, interface provider names.
        """
        litellm.drop_params = True

        self._tokens = 0

        if self.callbacks:
            self._set_callbacks(self.callbacks)

        self.context_window_size = self._get_context_window_size()

        base_url_key_name = self.endpoint_provider.upper() + "_API_BASE" if self.endpoint_provider else None
        if base_url_key_name:
            self.base_url = os.environ.get(base_url_key_name)
            self.api_base = self.base_url

        if self.llm_config:
            self._create_valid_params(config=self.llm_config)

        return self


    def _create_valid_params(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Returns valid params incl. model + litellm original params) from the given config dict.
        """
        valid_config, valid_keys = dict(), list()

        if self.model:
            valid_keys = get_supported_openai_params(model=self.model, custom_llm_provider=self.endpoint_provider, request_type="chat_completion")

        if not valid_keys:
            valid_keys = PARAMS.get("common")

        valid_keys += PARAMS.get("litellm")

        for item in valid_keys:
            if hasattr(self, item) and (getattr(self, item) or getattr(self, item) == False):
                valid_config[item] = getattr(self, item)

            elif item in self.llm_config and (self.llm_config[item] or self.llm_config[item]==False):
                valid_config[item] = self.llm_config[item]

            elif item in config and (config[item] or config[item] == False):
                valid_config[item] = config[item]

            else:
                pass

        if "max_tokens" in valid_config:
            if self.context_window_size and valid_config["max_tokens"] > self.context_window_size:
                valid_config["max_tokens"] = self.context_window_size

        self.llm_config = valid_config
        return valid_config


    def _set_env_vars(self) -> Dict[str, Any]:
        if self.provider == "openai":
            return {}

        cred = dict()
        env_vars = ENV_VARS.get(self.provider, None) if self.provider else None

        if not env_vars:
            return {}

        for item in env_vars:
            val = os.environ.get(item, None)
            if val:
                cred[str(item).lower()] = val

        return cred


    def _supports_function_calling(self) -> bool:
        try:
            if self.model:
                params = get_supported_openai_params(model=self.model)
                return "response_format" in params if params else False
            else:
                return False
        except Exception as e:
            self._logger.log(level="warning", message=f"Failed to get supported params: {str(e)}", color="yellow")
            return False


    def _supports_stop_words(self) -> bool:
        supported_params = get_supported_openai_params(model=self.model, custom_llm_provider=self.endpoint_provider)
        return "stop" in supported_params if supported_params else False


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
        litellm.set_verbose = True

        with suppress_warnings():
            if len(self.callbacks) > 0:
                self._set_callbacks(self.callbacks)

            try:
                res, tool_res = None, ""
                cred = self._set_env_vars()

                if self.provider == "gemini":
                    self.response_format = { "type": "json_object" } if not tools else None
                else:
                    self.response_format = response_format

                if not tools:
                    params = self._create_valid_params(config=config)
                    res = litellm.completion(model=self.model, messages=messages, stream=False, **params, **cred)
                    self._tokens += int(res["usage"]["total_tokens"])
                    return res["choices"][0]["message"]["content"]

                else:
                    try:
                        self.tools = [item.tool.properties if isinstance(item, ToolSet) else item.properties for item in tools]
                        params = self._create_valid_params(config=config)
                        res = litellm.completion(model=self.model, messages=messages, **params, **cred)
                        tool_calls = res.choices[0].message.tool_calls

                        if tool_calls:
                            for item in tool_calls:
                                func_name = item.function.name
                                func_args = item.function.arguments
                                if not isinstance(func_args, dict):
                                    try:
                                        func_args = json.loads(json.dumps(eval(str(func_args))))
                                    except:
                                        pass

                                # find a tool whose name is matched with the retrieved func_name
                                matches = []
                                for tool in tools:
                                    tool_name = tool.tool.name if isinstance(tool, ToolSet) else tool.name
                                    tool_func_name = tool.tool.func.__name__ if isinstance(tool, ToolSet) else tool.func.__name__
                                    if tool_name.replace(" ", "_") == func_name or tool_func_name == func_name or tool_name == "random_func":
                                        matches.append(tool)
                                    else:
                                        pass

                                if matches:
                                    tool_to_execute = matches[0]
                                    tool_instance = tool_to_execute.tool if isinstance(tool_to_execute, ToolSet) else tool_to_execute
                                    params = tool_to_execute.kwargs if isinstance(tool_to_execute, ToolSet) else func_args
                                    tool_res_to_add = tool_instance.run(params=params) if params else tool_instance.run()

                                    if tool_res_as_final:
                                        if tool_res_to_add not in tool_res:
                                            tool_res += str(tool_res_to_add)
                                    else:
                                        messages.append(res.choices[0].message)
                                        messages.append({ "role": "tool", "tool_call_id": item.id, "content": str(tool_res_to_add) })

                        else:
                            if tool_res_as_final and tools and not tool_res:
                                for item in tools:
                                    tool_res_to_add = item.tool.run(params=item.kwargs) if isinstance(item, ToolSet) else item.run()
                                    if tool_res_to_add not in tool_res:
                                        tool_res += str(tool_res_to_add)
                                    else:
                                        pass

                    except:
                        if tool_res_as_final and tools and not tool_res:
                            for item in tools:
                                tool_res_to_add = item.tool.run(params=item.kwargs) if isinstance(item, ToolSet) else item.run()
                                if tool_res_to_add not in tool_res:
                                    tool_res += str(tool_res_to_add)
                                else:
                                    pass
                        elif tools and not tool_res:
                                tool_res = res["choices"][0]["message"]["content"]
                        else:
                            pass


                if tool_res_as_final:
                    return tool_res
                else:
                    res = litellm.completion(model=self.model, messages=messages, **params, **cred)
                    self._tokens += int(res["usage"]["total_tokens"])
                    return res.choices[0].message.content


            except JSONSchemaValidationError as e:
                self._logger.log(level="error", message="Raw Response: {}".format(e.raw_response), color="red")
                raise e

            except Exception as e:
                self._logger.log(level="error", message=f"{self.model} failed to execute: {str(e)}", color="red")
                if "litellm.RateLimitError" in str(e):
                    raise e
