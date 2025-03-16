from typing import List, Dict, Any, Optional, Tuple

from versionhq.tool.gpt import openai_client
from versionhq.tool.gpt._enum import GPTFilterTypeEnum
from versionhq._utils import is_valid_enum, UsageMetrics, ErrorType


def is_valid_vector_store_id(id: str | list[str]) -> bool:
    if isinstance(id, list):
        for item in id:
            if not id.startswith("vs_"):
                return False
    else:
        return id.startswith("vs_")


class FilterSchema:
    class Filter:
        type: GPTFilterTypeEnum = GPTFilterTypeEnum.eq
        property: str = None
        value: str = None

        def __init__(self, **kwargs):
            for k, v in kwargs:
                if hasattr(self, k):
                    setattr(self, k, v)

        def _convert_to_schema(self) -> Dict[str, str] | None:
            return { "type": self.type, "property": self.property, "value": self.value } if self.property and self.value else None

    logic_type: str = "and"  # or
    filters: List[Filter] = list()
    filter_params: Dict[str, Any] = None

    def __init__(self, logic_type: str = None, filters: List[Filter] | Filter = None, filter_params: Dict[str, Any] = None, **kwargs):
        if logic_type == "and" | "or":
            self.logic_type = logic_type

        if filter_params:
            filter = FilterSchema.Filter()
            for k, v in filter_params.items():
                if k in FilterSchema.Filter.__dict__.keys():
                    if k == "type" and is_valid_enum(val=v, enum=GPTFilterTypeEnum):
                        setattr(filter, k, v if isinstance(v, str) else v.value)
                    else:
                        setattr(filter, k, v)
            self.filters.append(filter)

        if filters:
            match filters:
                case list():
                    if self.filters:
                        self.filters.extend(filters)
                    else:
                        self.filters = filters

                case FilterSchema.Filter():
                    if self.filters:
                        self.filters.append(filters)
                    else:
                        self.filters = [filters]

                case _:
                    pass

        if kwargs:
            for k, v in kwargs:
                if hasattr(self, k): setattr(self, k, v)

    @property
    def schema(self) -> Dict[str, Any] | None:
        if self.type and len(self.items) > 1:
            return {
                "type": self.type,
                "filters": [item._convert_to_schema() for item in self.items if isinstance(item, FilterSchema.Filter)]
            }
        elif self.filters:
            return self.filters[0]._convert_to_schema()
        else:
            return None


class GPTToolFileSearch:
    model: str = "gpt-4o"
    input: str = None
    vector_store_ids: List[str] = list()
    max_num_results: int = 2
    include: List[str] = ["output[*].file_search_call.search_results"]
    filters: Optional[FilterSchema] = None

    def __init__(
            self,
            input: str,
            vector_store_ids: str | List[str],
            model: str = None,
            max_num_results: int = None,
            include: List[str] = None,
            filters: FilterSchema | Dict[str, Any] = None,
            **kwargs,
        ):
        if not input or not vector_store_ids:
            return None

        if not is_valid_vector_store_id(id=vector_store_ids):
            return None

        self.input = input
        self.vector_store_ids = vector_store_ids if isinstance(vector_store_ids, list) else [vector_store_ids]
        self.model = model if model else self.model
        self.max_num_results = max_num_results if max_num_results else self.max_num_results
        self.include = include if include else self.include
        self.filters = filters if filters else None
        if kwargs:
            for k, v in kwargs.items():
                if hasattr(self, k):
                    setattr(self, k, v)


    def run(self) -> Tuple[str, List[Dict[str, Any]], UsageMetrics] | None:
        raw_res = ""
        annotations = list()
        usage = UsageMetrics()

        try:
            res = openai_client.responses.create(**self.schema)
            if not res:
                usage.record_errors(ErrorType.TOOL)
            else:
                raw_res = res.output[1].content[0].text
                annotations = [{ "index": item.index, "file_id": item.file_id, "filename": item.filename }
                                for item in res.output[1].content[0].annotations]
                usage.record_token_usage(**res.usage.__dict__)
            return raw_res, annotations, usage
        except:
            usage.record_errors(ErrorType.TOOL)
            return raw_res, annotations, usage


    @property
    def tool_schema(self) -> Dict[str, Any]:
        if self.filters:
            return [
                {
                    "type": "file_search",
                    "vector_store_ids": self.vector_store_ids,
                    "max_num_results": self.max_num_results,
                    "filters": self.filters.schema,
                }
            ]
        else:
            return [
                {
                    "type": "file_search",
                    "vector_store_ids": self.vector_store_ids,
                    "max_num_results": self.max_num_results,
                }
            ]


    @property
    def schema(self) -> Dict[str, Any]:
        schema = dict(model=self.model, tools=self.tool_schema, input=self.input)
        return schema
