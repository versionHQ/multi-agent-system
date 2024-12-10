import logging
import os
import sys
import threading
import warnings
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Union

import litellm
from dotenv import load_dotenv
from litellm import get_supported_openai_params

from components.llm.llm_vars import LLM_CONTEXT_WINDOW_SIZES
from components.task import OutputFormat
from components.task.model import ResponseField

load_dotenv(override=True)
API_KEY_LITELLM = os.environ.get("API_KEY_LITELLM")


class FilteredStream:
    def __init__(self, original_stream):
        self._original_stream = original_stream
        self._lock = threading.Lock()

    def write(self, s) -> int:
        with self._lock:
            if (
                "Give Feedback / Get Help: https://github.com/BerriAI/litellm/issues/new"
                in s
                or "LiteLLM.Info: If you need to debug this error, use `litellm.set_verbose=True`"
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


class LLM:
    """
    Use LiteLLM to connect with the model of choice.
    """

    def __init__(
        self,
        model: str,
        timeout: Optional[Union[float, int]] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        n: Optional[int] = None,
        stop: Optional[Union[str, List[str]]] = None,
        max_completion_tokens: Optional[int] = None,
        max_tokens: Optional[int] = None,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        logit_bias: Optional[Dict[int, float]] = None,
        # response_format: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
        logprobs: Optional[bool] = None,
        top_logprobs: Optional[int] = None,
        base_url: Optional[str] = None,
        api_version: Optional[str] = None,
        api_key: Optional[str] = None,
        callbacks: List[Any] = [],
        **kwargs,
    ):
        self.model = model
        self.timeout = timeout
        self.temperature = temperature
        self.top_p = top_p
        self.n = n
        self.stop = stop
        self.max_completion_tokens = max_completion_tokens
        self.max_tokens = max_tokens
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty
        self.logit_bias = logit_bias
        # self.response_format = response_format
        self.seed = seed
        self.logprobs = logprobs
        self.top_logprobs = top_logprobs
        self.base_url = base_url
        self.api_version = api_version
        self.api_key = api_key if api_key else API_KEY_LITELLM
        self.callbacks = callbacks
        self.kwargs = kwargs

        litellm.drop_params = True
        litellm.set_verbose = True
        self.set_callbacks(callbacks)

    def call(
        self,
        output_formats: List[OutputFormat],
        field_list: Optional[List[ResponseField]],
        messages: List[Dict[str, str]],
        callbacks: List[Any] = [],
    ) -> str:

        with suppress_warnings():
            if callbacks and len(callbacks) > 0:
                self.set_callbacks(callbacks)

            try:
                response_format = None

                if OutputFormat.JSON in output_formats:
                    response_format = LLMResponseSchema(
                        response_type="json_object", field_list=field_list
                    )

                params = {
                    "model": self.model,
                    "messages": messages,
                    "timeout": self.timeout,
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "n": self.n,
                    "stop": self.stop,
                    "max_tokens": self.max_tokens or self.max_completion_tokens,
                    "presence_penalty": self.presence_penalty,
                    "frequency_penalty": self.frequency_penalty,
                    "logit_bias": self.logit_bias,
                    # "response_format": response_format,
                    "seed": self.seed,
                    "logprobs": self.logprobs,
                    "top_logprobs": self.top_logprobs,
                    "api_base": self.base_url,
                    "api_version": self.api_version,
                    "api_key": self.api_key,
                    "stream": False,
                    **self.kwargs,
                }
                params = {k: v for k, v in params.items() if v is not None}
                res = litellm.completion(**params)
                return res["choices"][0]["message"]["content"]

            except Exception as e:
                logging.error(f"LiteLLM call failed: {str(e)}")
                raise

    def supports_function_calling(self) -> bool:
        try:
            params = get_supported_openai_params(model=self.model)
            return "response_format" in params
        except Exception as e:
            logging.error(f"Failed to get supported params: {str(e)}")
            return False

    def supports_stop_words(self) -> bool:
        try:
            params = get_supported_openai_params(model=self.model)
            return "stop" in params
        except Exception as e:
            logging.error(f"Failed to get supported params: {str(e)}")
            return False

    def get_context_window_size(self) -> int:
        # Only using 75% of the context window size to avoid cutting the message in the middle
        return int(LLM_CONTEXT_WINDOW_SIZES.get(self.model, 8192) * 0.75)

    def set_callbacks(self, callbacks: List[Any]):
        callback_types = [type(callback) for callback in callbacks]
        for callback in litellm.success_callback[:]:
            if type(callback) in callback_types:
                litellm.success_callback.remove(callback)

        for callback in litellm._async_success_callback[:]:
            if type(callback) in callback_types:
                litellm._async_success_callback.remove(callback)

        litellm.callbacks = callbacks
