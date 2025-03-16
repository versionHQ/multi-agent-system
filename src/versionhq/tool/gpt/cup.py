from typing import List, Dict, Any

from versionhq._utils import convert_img_url
from versionhq.tool.gpt import openai_client
from versionhq.tool.gpt._enum import GPTCUPEnvironmentEnum, GPTCUPTypeEnum, GPTSizeEnum
from versionhq._utils import is_valid_enum, UsageMetrics, ErrorType


class CUPToolSchema:
    type: str = GPTCUPTypeEnum.COMPUTER_USE_PREVIEW.value
    display_width: int = 1024
    display_height: int = 768
    environment: str = GPTCUPEnvironmentEnum.BROWSER.value

    def __init__(
            self,
            type: str | GPTCUPTypeEnum = None,
            display_width: int = None,
            display_height: int = None,
            environment: str | GPTCUPEnvironmentEnum = None
        ):
        self.display_height = display_height if display_height else self.display_height
        self.display_width = display_width if display_width else self.display_width

        if type and is_valid_enum(enum=GPTCUPTypeEnum, val=type):
            self.type = type.value if isinstance(type, GPTCUPTypeEnum) else type

        if environment and is_valid_enum(enum=GPTCUPEnvironmentEnum, val=environment):
            self.environment = environment.value if isinstance(environment, GPTCUPEnvironmentEnum) else environment

        self.environment = environment if environment else self.environment


    @property
    def schema(self) -> Dict[str, Any]:
        return {
            "type": self.type if isinstance(self.type, str) else self.type.value,
            "display_width": self.display_width,
            "display_height": self.display_height,
            "environment": self.environment if isinstance(self.environment, str) else self.environment.value,
        }


class GPTToolCUP:
    model: str = "computer-use-preview"
    tools: List[CUPToolSchema] = list()
    user_prompt: str = None
    img_url: str = None
    reasoning_effort: str = GPTSizeEnum.MEDIUM.value
    truncation: str = "auto"

    def __init__(
        self,
        user_prompt: str,
        tools: List[CUPToolSchema] | CUPToolSchema = None,
        img_url: str = None,
        reasoning_effort: GPTSizeEnum | str = None,
        truncation: str = None
    ):
        self.user_prompt = user_prompt
        self.truncation = truncation if truncation else self.truncation

        if img_url:
            img_url = convert_img_url(img_url)
            self.img_url = img_url

        if reasoning_effort and is_valid_enum(enum=GPTSizeEnum, val=reasoning_effort):
            self.reasoning_effort = reasoning_effort.value if isinstance(reasoning_effort, GPTSizeEnum) else reasoning_effort

        if tools:
            match tools:
                case list():
                    if self.tools:
                        self.tools.extend(tools)
                    else:
                        self.tools = tools
                case CUPToolSchema():
                    if self.tools:
                        self.tools.append(tools)
                    else:
                        self.tools = [tools]
                case _:
                    pass

    def run(self):
        raw_res = ""
        usage = UsageMetrics()

        try:
            res = openai_client.responses.create(**self.schema)
            if not res:
                usage.record_errors(ErrorType.TOOL)
            else:
                raw_res = res.output[1].summary[0].text
                usage.record_token_usage(**res.usage.__dict__)
            return raw_res, None, usage
        except:
            usage.record_errors(ErrorType.TOOL)
            return raw_res, None, usage


    @property
    def schema(self) -> Dict[str, Any]:
        img_url = convert_img_url(self.img_url)  if self.img_url else None
        inputs = [{ "role": "user", "content": self.user_prompt } ]

        if img_url:
            inputs.append({"type": "input_image", "image_url": f"data:image/png;base64,{img_url}"})

        tool_schema = [item.schema for item in self.tools]
        schema = dict(model=self.model, tools=tool_schema, input=inputs, reasoning={ "effort": self.reasoning_effort}, truncation=self.truncation)
        return schema


#     "output": [
#     {
#         "type": "reasoning",
#         "id": "rs_67cb...",
#         "summary": [
#             {
#                 "type": "summary_text",
#                 "text": "Exploring 'File' menu option."
#             }
#         ]
#     },
#     {
#         "type": "computer_call",
#         "id": "cu_67cb...",
#         "call_id": "call_nEJ...",
#         "action": {
#             "type": "click",
#             "button": "left",
#             "x": 135,
#             "y": 193
#         },
#         "pending_safety_checks": [
#             {
#                 "id": "cu_sc_67cb...",
#                 "code": "malicious_instructions",
#                 "message": "We've detected instructions that may cause your application to perform malicious or unauthorized actions. Please acknowledge this warning if you'd like to proceed."
#             }
#         ],
#         "status": "completed"
#     }
# ]
