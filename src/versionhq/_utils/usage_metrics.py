from pydantic import BaseModel, Field


class UsageMetrics(BaseModel):
    """
    Model to track usage
    """

    total_tokens: int = Field(default=0, description="total number of tokens used")
    prompt_tokens: int = Field(default=0, description="number of tokens used in prompts")
    cached_prompt_tokens: int = Field(default=0, description="number of cached prompt tokens used")
    completion_tokens: int = Field(default=0, description="number of tokens used in completions")
    successful_requests: int = Field(default=0, description="number of successful requests made")

    def add_usage_metrics(self, usage_metrics: "UsageMetrics") -> None:
        """
        Add the usage metrics from another UsageMetrics object.
        """
        self.total_tokens += usage_metrics.total_tokens
        self.prompt_tokens += usage_metrics.prompt_tokens
        self.cached_prompt_tokens += usage_metrics.cached_prompt_tokens
        self.completion_tokens += usage_metrics.completion_tokens
        self.successful_requests += usage_metrics.successful_requests



# class TokenProcess:
#     total_tokens: int = 0
#     prompt_tokens: int = 0
#     cached_prompt_tokens: int = 0
#     completion_tokens: int = 0
#     successful_requests: int = 0

#     def sum_prompt_tokens(self, tokens: int) -> None:
#         self.prompt_tokens = self.prompt_tokens + tokens
#         self.total_tokens = self.total_tokens + tokens

#     def sum_completion_tokens(self, tokens: int) -> None:
#         self.completion_tokens = self.completion_tokens + tokens
#         self.total_tokens = self.total_tokens + tokens

#     def sum_cached_prompt_tokens(self, tokens: int) -> None:
#         self.cached_prompt_tokens = self.cached_prompt_tokens + tokens

#     def sum_successful_requests(self, requests: int) -> None:
#         self.successful_requests = self.successful_requests + requests

#     def get_summary(self) -> UsageMetrics:
#         return UsageMetrics(
#             total_tokens=self.total_tokens,
#             prompt_tokens=self.prompt_tokens,
#             cached_prompt_tokens=self.cached_prompt_tokens,
#             completion_tokens=self.completion_tokens,
#             successful_requests=self.successful_requests,
#         )
