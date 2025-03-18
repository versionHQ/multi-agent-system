import datetime
import time
from typing import List, Dict, Any, Tuple

from versionhq._utils import convert_img_url
from versionhq.tool.gpt import openai_client
from versionhq.tool.gpt._enum import GPTCUAEnvironmentEnum, GPTCUATypeEnum, GPTSizeEnum
from versionhq._utils import is_valid_enum, UsageMetrics, ErrorType, Logger, is_valid_url


allowed_browsers = ['webkit', 'chromium', 'firefox']


class CUAToolSchema:
    type: str = GPTCUATypeEnum.COMPUTER_USE_PREVIEW.value
    display_width: int = 1024
    display_height: int = 768
    environment: str = GPTCUAEnvironmentEnum.BROWSER.value

    def __init__(
            self,
            type: str | GPTCUATypeEnum = None,
            display_width: int = None,
            display_height: int = None,
            environment: str | GPTCUAEnvironmentEnum = None
        ):
        self.display_height = display_height if display_height else self.display_height
        self.display_width = display_width if display_width else self.display_width

        if type and is_valid_enum(enum=GPTCUATypeEnum, val=type):
            self.type = type.value if isinstance(type, GPTCUATypeEnum) else type

        if environment and is_valid_enum(enum=GPTCUAEnvironmentEnum, val=environment):
            self.environment = environment.value if isinstance(environment, GPTCUAEnvironmentEnum) else environment

        self.environment = environment if environment else self.environment


    @property
    def schema(self) -> Dict[str, Any]:
        return {
            "type": self.type if isinstance(self.type, str) else self.type.value,
            "display_width": self.display_width,
            "display_height": self.display_height,
            "environment": self.environment if isinstance(self.environment, str) else self.environment.value,
        }


class GPTToolCUA:
    model: str = "computer-use-preview"
    tools: List[CUAToolSchema] = list()
    user_prompt: str = None
    img_url: str = None
    web_url: str = "https://www.google.com"
    browser: str = "firefox"
    reasoning_effort: str = GPTSizeEnum.MEDIUM.value
    truncation: str = "auto"

    _response_ids: List[str] = list()
    _call_ids: List[str] = list()
    _usage: UsageMetrics = UsageMetrics()
    _logger: Logger = Logger(info_file_save=True, filename="cua-task-{}".format(str(datetime.datetime.now().timestamp())) + ".png")


    def __init__(
        self,
        user_prompt: str,
        tools: List[CUAToolSchema] | CUAToolSchema = None,
        img_url: str = None,
        web_url: str = "https://www.google.com",
        browser: str = "chromium",
        reasoning_effort: GPTSizeEnum | str = None,
        truncation: str = None,
        _usage: UsageMetrics = UsageMetrics()
    ):
        self.user_prompt = user_prompt
        self.web_url = web_url if is_valid_url(web_url) else "https://www.google.com"
        self.browser = browser if browser in allowed_browsers else 'chromium'
        self.truncation = truncation if truncation else self.truncation
        self._usage = _usage
        self._response_ids = list()
        self._call_ids = list()

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
                case CUAToolSchema():
                    if self.tools:
                        self.tools.append(tools)
                    else:
                        self.tools = [tools]
                case _:
                    pass


    def _take_screenshot(self, page: Any = None, path: str = None) -> Tuple[str | None, str | None]:
        import base64
        if not page:
            return None, None

        path = path if path else "screenshot.png"
        screenshot_bytes = page.screenshot()
        screenshot_base64 = base64.b64encode(screenshot_bytes).decode("utf-8")
        self._logger.log(message=f"Action: screenshot", level="info", color="blue")
        return screenshot_bytes, screenshot_base64


    def _handle_model_action(self, page: Any, action: Any, action_type: str = None) -> bool:
        """Creates a page object and performs actions."""

        action_type = action_type if action_type else action.type
        start_dt = datetime.datetime.now()

        try:
            match action_type:
                case "click":
                    x, y = action.x, action.y
                    button = action.button
                    self._logger.log(message=f"Action: click at ({x}, {y}) with button '{button}'", level="info", color="blue")
                    if button != "left" and button != "right":
                        button = "left"
                    page.mouse.click(x, y, button=button)

                case "scroll":
                    x, y = action.x, action.y
                    scroll_x, scroll_y = action.scroll_x, action.scroll_y
                    self._logger.log(message=f"Action: scroll at ({x}, {y}) with offsets (scroll_x={scroll_x}, scroll_y={scroll_y})", level="info", color="blue")
                    page.mouse.move(x, y)
                    page.evaluate(f"window.scrollBy({scroll_x}, {scroll_y})")

                case "keypress":
                    keys = action.keys
                    for k in keys:
                        self._logger.log(message=f"Action: keypress '{k}'", level="info", color="blue")
                        if k.lower() == "enter":
                            page.keyboard.press("Enter")
                        elif k.lower() == "space":
                            page.keyboard.press(" ")
                        else:
                            page.keyboard.press(k)

                case "type":
                    text = action.text
                    self._logger.log(message=f"Action: type text: {text}", level="info", color="blue")
                    page.keyboard.type(text)

                case "wait":
                    self._logger.log(message=f"Action: wait", level="info", color="blue")
                    time.sleep(2)

                case "screenshot":
                    pass

                case _:
                    self._logger.log(message=f"Unrecognized action: {action}", level="warning", color="yellow")

        except Exception as e:
            self._usage.record_errors(type=ErrorType.API)
            self._logger.log(message=f"Error handling action {action}: {e}", level="error", color="red")

        end_dt = datetime.datetime.now()
        self._usage.record_latency(start_dt=start_dt, end_dt=end_dt)
        return bool(self._usage.total_errors)


    def run(self, screenshot: str = None) -> Tuple[Dict[str, Any], None, UsageMetrics]:
        raw_res = dict()
        usage = self._usage if self._usage else UsageMetrics()
        start_dt = datetime.datetime.now()

        try:
            schema = self.schema
            if screenshot and "output" in schema["input"][0]:
                output_image_url = schema["input"][0]["output"]["image_url"].replace("SCREENSHOT", str(screenshot))
                schema["input"][0]["output"]["image_url"] = output_image_url

            res = openai_client.responses.create(**schema)
            if not res:
                usage.record_errors(ErrorType.TOOL)
            else:
                for item in res.output:
                    match item.type:
                        case "reasoning":
                            raw_res.update(dict(reasoning=item.summary[0].text))
                            if item.id and item.id.startwith('rs'):
                                self._response_ids.append(item.id)
                        case "computer_call":
                            raw_res.update(dict(action=item.action))
                            # self._response_ids.append(item.id)
                            self._call_ids.append(item.call_id)
                        case _:
                            pass
                usage.record_token_usage(**res.usage.__dict__)

        except Exception as e:
            self._logger.log(message=f"Failed to run: {str(e)}", color="red", level="error")
            usage.record_errors(ErrorType.TOOL)

        end_dt = datetime.datetime.now()
        usage.record_latency(start_dt=start_dt, end_dt=end_dt)
        return raw_res, None, usage


    def invoke_playwright(self) -> Tuple[Dict[str, Any], None, UsageMetrics]:
        """Handles computer use loop. Ref. OpenAI official website."""

        from playwright.sync_api import sync_playwright

        self._logger.log(message="Start the operation.", level="info", color="blue")

        try:
            with sync_playwright() as p:
                b = p.firefox if self.browser == "firefox" else p.webkit if self.browser == "webkit" else p.chromium
                browser = b.launch(headless=True)
                page = browser.new_page()
                if not browser or not page:
                    return None, None, None

                page.goto(self.web_url)
                res, _, usage = self.run()
                self._usage = usage
                actions = [v for k, v in res.items() if k =="action"] if res else []
                action = actions[0] if actions else None
                start_dt = datetime.datetime.now()

                if action:
                    while True:
                        self._handle_model_action(page=page, action=action)
                        _, screenshot_base64 = self._take_screenshot(page=page)
                        res, _, usage = self.run(screenshot=screenshot_base64)
                        self._usage.agggregate(metrics=usage)
                        if not res:
                            usage.record_errors(type=ErrorType.API)
                            break

                        actions = [v for k, v in res.items() if k =="action"] if res else []
                        action = actions[0] if actions else None
                        if not action:
                            break
                else:
                    self._usage.record_errors(type=ErrorType.TOOL)

        except Exception as e:
            self._logger.log(message=f"Failed to execute. {str(e)}", color="red", level="error")

        end_dt = datetime.datetime.now()
        self._usage.record_latency(start_dt=start_dt, end_dt=end_dt)
        # browser.close()
        return res, _, self._usage


    @property
    def schema(self) -> Dict[str, Any]:
        """Formats args schema for CUA calling."""

        tool_schema = [item.schema for item in self.tools]
        schema = dict()
        inputs = list()
        previous_response_id = self._response_ids[-1] if self._response_ids and self._response_ids[-1].startswith("rs") else None

        if self._call_ids:
            inputs = [
                {
                    "call_id": self._call_ids[-1],
                    "type": "computer_call_output",
                    "output": { "type": "input_image", "image_url": f"data:image/png;base64,SCREENSHOT"}
                }
            ]
            schema = dict(
                model=self.model,
                previous_response_id=previous_response_id,
                tools=tool_schema,
                input=inputs,
                truncation=self.truncation
            )

        else:
            img_url = convert_img_url(self.img_url) if self.img_url else None
            input = [{ "role": "user", "content": self.user_prompt } ]
            if img_url:
                input.append({"type": "input_image", "image_url": f"data:image/png;base64,{img_url}"})
            schema = dict(model=self.model, tools=tool_schema, input=input, reasoning={ "effort": self.reasoning_effort}, truncation=self.truncation)

        return schema
