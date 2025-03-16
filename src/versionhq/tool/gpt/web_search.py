from typing import Dict, Any, Optional, Tuple, List

from versionhq.tool.gpt import openai_client
from versionhq.tool.gpt._enum import GPTSizeEnum
from versionhq._utils import is_valid_enum, UsageMetrics, ErrorType


class GPTToolWebSearch:
    """A class to manage Web Search tools by OpenAI."""

    model: str = "gpt-4o"
    input: str = None
    location_type: str = None  # "approximate"
    country: str = None  # "GB"
    city: str = None # "London"
    region: str = None # "London"
    search_content_size: str = GPTSizeEnum.MEDIUM.value
    _user_location: Optional[Dict[str, str]] = None


    def __init__(
            self,
            model: str = None,
            input: str = None,
            location_type: str = None,
            country: str = None,
            city: str = None,
            region: str = None,
            search_content_size = str | GPTSizeEnum,
            **kwargs,
        ):
        self.model = model if model else self.model
        self.input = input if input else self.input
        if country and city and region:
            self.location_type = location_type if location_type else "approximate"
            self.country = country
            self.city = city
            self.region = region
            self._user_location = dict(type=self.location_type, country=self.country, city=self.city, region=self.region)

        if is_valid_enum(val=search_content_size, enum=GPTSizeEnum):
            self.search_content_size = search_content_size if isinstance(search_content_size, str) else search_content_size.value

        if kwargs:
            for k, v in kwargs.items():
                if hasattr(self, k):
                    setattr(self, k, v)


    def run(self) -> Tuple[str, List[Dict[str, str]], UsageMetrics]:
        """Runs the tool and returns text response, annotations, and usage metrics."""

        raw_res = ""
        annotations = list()
        usage = UsageMetrics()

        try:
            res = openai_client.responses.create(**self.schema)
            if not res:
                usage.record_errors(ErrorType.TOOL)
            else:
                raw_res = res.output[1].content[0].text
                annotations = [{ "title": item.title, "url": item.url } for item in res.output[1].content[0].annotations]
                usage.record_token_usage(**res.usage.__dict__)
            return raw_res, annotations, usage
        except:
            usage.record_errors(ErrorType.TOOL)
            return raw_res, annotations, usage


    @property
    def tool_schema(self) -> Dict[str, Any]:
        if self._user_location:
            return {
                "type": "web_search_preview",
                "user_location": self._user_location,
                "search_context_size": self.search_content_size
            }
        else:
            return {
                "type": "web_search_preview",
                "search_context_size": self.search_content_size
            }


    @property
    def schema(self) -> Dict[str, Any]:
        schema = dict(model=self.model, tools=[self.tool_schema,], input=self.input)
        return schema
