import logging
import os
import sys
import threading
import warnings
import litellm
from abc import ABC
from dotenv import load_dotenv
from litellm import get_supported_openai_params
from contextlib import contextmanager
from typing import Any, Dict, List, Optional
from typing_extensions import Self

from pydantic import UUID4, BaseModel, Field, PrivateAttr, field_validator, model_validator, create_model, InstanceOf, ConfigDict
from pydantic_core import PydanticCustomError

from versionhq.llm.llm_variables import LLM_CONTEXT_WINDOW_SIZES, LLM_API_KEY_NAMES, LLM_BASE_URL_KEY_NAMES, MODELS, LITELLM_COMPLETION_KEYS
from versionhq.task import TaskOutputFormat
from versionhq.task.model import ResponseField
from versionhq._utils.logger import Logger


load_dotenv(override=True)
API_KEY_LITELLM = os.environ.get("API_KEY_LITELLM")
DEFAULT_CONTEXT_WINDOW_SIZE = int(8192 * 0.75)
DEFAULT_MODEL_NAME = os.environ.get("DEFAULT_MODEL_NAME")

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


class LLMResponseSchema:
    """
    Use the response schema for LLM response.
    `field_list` contains the title, value type, bool if required of each field that needs to be returned.
    field_list: [{ title, type, required } ]

    i.e., reponse_schema
    response_type: "array"  *options: "array", "dict"
    propeties: { "recipe_name": { "type": "string" }, },
    required:  ["recipe_name"]
    """

    def __init__(self, response_type: str, field_list: List[ResponseField]):
        self.type = response_type
        self.field_list = field_list

    @property
    def schema(self):
        if len(self.field_list) == 0:
            return

        properties = [
            {
                field.title: {
                    "type": field.type,
                }
            }
            for field in self.field_list
        ]
        required = [field.title for field in self.field_list if field.required == True]
        response_schema = {
            "type": self.type,
            "items": {"type": "object", "properties": {*properties}},
            "required": required,
        }
        return response_schema


class LLM(BaseModel):
    """
    An LLM class to store params except for response formats which will be given in the task handling process.
    Use LiteLLM to connect with the model of choice.
    Some optional params are passed by the agent, else follow the default settings of the model provider.
    """

    _logger: Logger = PrivateAttr(default_factory=lambda: Logger(verbose=True))
    _init_model_name: str = PrivateAttr(default=None)
    model_config = ConfigDict(extra="allow")

    model: str = Field(default=DEFAULT_MODEL_NAME)
    provider: Optional[str] = Field(default=None, description="model provider or custom model provider")
    base_url: Optional[str] = Field(default=None, description="litellm's api base")
    api_key: Optional[str] = Field(default=None)
    api_version: Optional[str] = Field(default=None)

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

    litellm.drop_params = True
    litellm.set_verbose = True
    os.environ['LITELLM_LOG'] = 'DEBUG'


    @model_validator(mode="after")
    def validate_base_params(self) -> Self:
        """
        1. Model name and provider
        Check the provided model name in the list and update it with the valid model key name.
        Then add the model provider if it is not provided.
        Assign a default model and provider when we cannot find a model key.

        2.  Set up other base parameters for the model and LiteLLM as below:
        1. LiteLLM - drop_params, set_verbose, callbacks
        2. Model setup - context_window_size, api_key, base_url
        """

        if self.model is None:
            self._logger.log(level="error", message="Model name is missing.", color="red")
            raise PydanticCustomError("model_missing", "The model name must be provided.", {})


        self._init_model_name = self.model
        self.model = None

        if self.provider and MODELS.get(self.provider) is not None:
            provider_model_list = MODELS.get(self.provider)
            for item in provider_model_list:
                if self.model is None:
                    if item == self._init_model_name:
                        self.model = item
                    elif self._init_model_name in item and self.model is None:
                        self.model = item
                    else:
                        temp_model = provider_model_list[0]
                        self._logger.log(level="info", message=f"The provided model: {self._init_model_name} is not in the list. We'll assign a model: {temp_model} from the selected model provider: {self.provider}.", color="yellow")
                        self.model = temp_model
                    # raise PydanticCustomError("invalid_model", "The provided model is not in the list.", {})

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
                self._logger.log(level="info", message=f"The provided model \'{self.model}\' is not in the list. We'll assign a default model.", color="yellow")
                self.model = DEFAULT_MODEL_NAME
                self.provider = "openai"
                # raise PydanticCustomError("invalid_model", "The provided model is not in the list.", {})

        if self.callbacks is not None:
            self._set_callbacks(self.callbacks)

        self.context_window_size = self._get_context_window_size()

        api_key_name = LLM_API_KEY_NAMES.get(self.provider, "LITELLM_API_KEY")
        self.api_key = os.environ.get(api_key_name, None)

        base_url_key_name = LLM_BASE_URL_KEY_NAMES.get(self.provider, "OPENAI_API_BASE")
        self.base_url = os.environ.get(base_url_key_name, None)

        return self


    def call(
        self,
        output_formats: List[str | TaskOutputFormat],
        field_list: Optional[List[ResponseField]],
        messages: List[Dict[str, str]],
        **kwargs,
        # callbacks: List[Any] = [],
    ) -> str:
        """
        Execute LLM based on the agent's params and model params.
        """

        with suppress_warnings():
            if len(self.callbacks) > 0:
                self._set_callbacks(self.callbacks)

            try:
                # response_format = None
                # #! REFINEME
                # if TaskOutputFormat.JSON in output_formats:
                #     response_format = LLMResponseSchema(
                #         response_type="json_object", field_list=field_list
                #     )

                params = {}
                for item in LITELLM_COMPLETION_KEYS:
                    if hasattr(self, item) and getattr(self, item) is not None:
                        params[item] = getattr(self, item)

                res = litellm.completion(messages=messages, stream=False, **params)
                return res["choices"][0]["message"]["content"]

            except Exception as e:
                self._logger.log(level="error", message=f"LiteLLM call failed: {str(e)}", color="red")
                return None


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
