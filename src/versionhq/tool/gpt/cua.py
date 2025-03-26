import base64
import datetime
import time
import platform
from typing import List, Dict, Any, Tuple

from versionhq._utils import convert_img_url
from versionhq.tool.gpt import openai_client
from versionhq.tool.gpt._enum import GPTCUABrowserEnum, GPTCUATypeEnum, GPTSizeEnum
from versionhq._utils import is_valid_enum, UsageMetrics, ErrorType, Logger, is_valid_url, handle_directory

allowed_browsers = ['chromium', 'firefox']


class CUAToolSchema:
    type: str = GPTCUATypeEnum.COMPUTER_USE_PREVIEW.value
    display_width: int = 1024
    display_height: int = 768
    environment: str = GPTCUABrowserEnum.BROWSER.value

    def __init__(
            self,
            type: str | GPTCUATypeEnum = None,
            display_width: int = None,
            display_height: int = None,
            environment: str | GPTCUABrowserEnum = None
        ):
        self.display_height = display_height if display_height else self.display_height
        self.display_width = display_width if display_width else self.display_width

        if type and is_valid_enum(enum=GPTCUATypeEnum, val=type):
            self.type = type.value if isinstance(type, GPTCUATypeEnum) else type

        if environment and is_valid_enum(enum=GPTCUABrowserEnum, val=environment):
            self.environment = environment.value if isinstance(environment, GPTCUABrowserEnum) else environment

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

    _schema: Dict[str, Any] = dict()
    _response_ids: List[str] = list()
    _call_ids: List[str] = list()
    _calls: Dict[str, Dict[str, Any]] = dict() # stores response_id and raw output object.
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
        self.web_url = web_url if is_valid_url(web_url) else None
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


    def _structure_schema(self, screenshot: str = None) -> None:
        """Formats args schema for CUA calling."""

        tool_schema = [item.schema for item in self.tools]
        schema = dict()
        inputs = list()
        previous_response_id = self._response_ids[-1] if self._response_ids else None
        # (self._response_ids[-1].startswith("rs") or  self._response_ids[-1].startswith("resp")) else None

        if self._call_ids:
            inputs = [
                {
                    "call_id": self._call_ids[-1],
                    "type": "computer_call_output",
                }
            ]
            if screenshot:
                inputs[0].update({ "output": { "type": "computer_screenshot", "image_url": f"data:image/png;base64,{str(screenshot)}"}})

            # if self._calls:
            #     call = self._calls[self._call_ids[-1]]
            #     if call and call.call_id not in inputs[0]:
            #         inputs.append(call)

            if previous_response_id:
                schema = dict(
                    model=self.model,
                    previous_response_id=previous_response_id,
                    tools=tool_schema,
                    input=inputs,
                    truncation=self.truncation
                )
            else:
                schema = dict(
                    model=self.model,
                    tools=tool_schema,
                    input=inputs,
                    truncation=self.truncation
                )

        else:
            input = [{ "role": "user", "content": self.user_prompt } ]
            img_url = convert_img_url(self.img_url) if self.img_url else None
            if img_url:
                input.append({"type": "input_image", "image_url": f"data:image/png;base64,{img_url}"})

            schema = dict(
                model=self.model,
                tools=tool_schema,
                input=input,
                reasoning={ "effort": self.reasoning_effort},
                truncation=self.truncation
            )

        self._schema = schema
        # return self._schema


    def _run(self, screenshot: str = None) -> Tuple[Dict[str, Any], None, UsageMetrics]:
        raw_res = dict()
        usage = self._usage if self._usage else UsageMetrics()
        start_dt = datetime.datetime.now()

        try:
            self._structure_schema(screenshot=screenshot)
            res = openai_client.responses.create(**self._schema)
            if not res:
                usage.record_errors(ErrorType.TOOL)
            else:
                self._response_ids.append(res.id)
                for item in res.output:

                    match item.type:
                        case "reasoning":
                            reasoning = item.summary[0].text if item.summary and isinstance(item.summary, list) else str(item.summary) if item.summary else ""
                            raw_res.update(dict(reasoning=reasoning))
                            # self._response_ids.append(item.id)

                        case "computer_call":
                            raw_res.update(dict(action=item.action))
                            # self._response_ids.append(item.id)
                            call_id = item.call_id
                            self._call_ids.append(call_id)
                            self._calls.update({ call_id: item })
                        case _:
                            pass
            usage.record_token_usage(**res.usage.__dict__)

        except Exception as e:
            self._logger.log(message=f"Failed to run: {str(e)}", color="red", level="error")
            usage.record_errors(ErrorType.TOOL)

        end_dt = datetime.datetime.now()
        usage.record_latency(start_dt=start_dt, end_dt=end_dt)
        return raw_res, None, usage


    def invoke_playwright(self) -> Dict[str, Any]:
        """Handles computer use loop. Ref. OpenAI official website."""
        try:
            from playwright.sync_api import sync_playwright
        except Exception as e:
            self._logger.log(level="error", message=f"Install Playwright by adding `versionhq[tools]` to requirements.txt or run `uv add playwright`. {str(e)}", color="red")
            raise e

        import os
        os.environ["DEBUG"] = "pw:browser"
        self._logger.log(message="Start computer use.", level="info", color="blue")
        start_dt = datetime.datetime.now()
        res = None

        # try:
        p = sync_playwright().start()
        b = p.firefox if self.browser == "firefox" else p.chromium
        browser = b.launch(headless=True)
        page = browser.new_page()
        if not browser or not page:
            return None, None, None

        if self.web_url:
            page.goto(self.web_url, timeout=3000000, wait_until="load", referer=None)
            time.sleep(3)

        res, _, usage = self._run()
        self._usage.aggregate(metrics=usage)
        actions = [v for k, v in res.items() if k =="action"] if res else []
        action = actions[0] if actions else None

        if action:
            while True:
                x = action.x if hasattr(action, 'x') else 0
                y = action.y if hasattr(action, 'y') else 0
                scroll_x = action.scroll_x if hasattr(action, 'scroll_x') else 0
                scroll_y = action.scroll_y if hasattr(action, 'scroll_y') else 0
                text = action.text if hasattr(action, 'text') else ''
                screenshot_base64 = None
                path = handle_directory(directory_name='_screenshots', filename=f'cua_playwright', ext='png')

                match action.type:
                    case "click":
                        self._logger.log(message="Action: click", color="blue", level="info")
                        button = action.button if hasattr(action, 'button') and (action.button == 'left' or action.button == 'right') else 'left'
                        page.mouse.move(x, y)
                        page.mouse.click(x, y, button=button)
                        time.sleep(1)

                    case "scroll":
                        self._logger.log(message="Action: scroll", color="blue", level="info")
                        page.mouse.move(x, y)
                        page.evaluate(f"window.scrollBy({scroll_x}, {scroll_y})")
                        time.sleep(1)

                    case "move":
                        self._logger.log(message="Action: move", color="blue", level="info")
                        page.mouse.move(x, y)
                        page.evaluate(f"window.scrollBy({scroll_x}, {scroll_y})")
                        time.sleep(1)

                    case "keypress":
                        self._logger.log(message="Action: keypress", color="blue", level="info")
                        keys = action.keys
                        for k in keys:
                            match k.lower():
                                case "enter": page.keyboard.press("Enter")
                                case "space": page.keyboard.press(" ")
                                case _: page.keyboard.press(k)
                        time.sleep(1)

                    case "type":
                        self._logger.log(message="Action: type", color="blue", level="info")
                        page.keyboard.type(text)
                        time.sleep(1)

                    case "wait":
                        self._logger.log(message="Action: wait", color="blue", level="info")
                        time.sleep(3)

                    case "screenshot":
                        self._logger.log(message="Action: screenshot", color="blue", level="info")
                        screenshot_bytes = page.screenshot(path=path)
                        screenshot_base64 = base64.b64encode(screenshot_bytes).decode("utf-8")
                        time.sleep(1)

                    case _:
                        self._logger.log(message=f"Unrecognized action: {action}", level="warning", color="yellow")
                        return False

                if not screenshot_base64:
                    screenshot_bytes = page.screenshot(path=path)
                    screenshot_base64 = base64.b64encode(screenshot_bytes).decode("utf-8")
                    time.sleep(1)

                res, _, usage = self._run(screenshot=screenshot_base64)
                self._usage.aggregate(metrics=usage)
                if not res:
                    usage.record_errors(type=ErrorType.API)
                    break

                actions = [v for k, v in res.items() if k =="action"] if res else []
                action = actions[0] if actions else None
                if not action:
                    break
        else:
            self._usage.record_errors(type=ErrorType.TOOL)

        # except Exception as e:
        #     self._logger.log(message=f"Failed to execute. {str(e)}", color="red", level="error")
        #     browser.close()

        end_dt = datetime.datetime.now()
        self._usage.record_latency(start_dt=start_dt, end_dt=end_dt)
        return res


    def invoke_selenium(self, **kwargs) -> Dict[str, Any]:
        try:
            from selenium import webdriver
            from selenium.webdriver.common.keys import Keys
            from selenium.webdriver.common.action_chains import ActionChains
            from selenium.webdriver.common.actions.action_builder import ActionBuilder
        except Exception as e:
            self._logger.log(level="error", message=f"Install Selenium by `uv pip install versionhq[tools]` or `uv add selenium`. {str(e)}", color="red")
            raise e

        self._logger.log(message="Start computer use", level="info", color="blue")

        start_dt = datetime.datetime.now()

        driver = webdriver.Chrome(options=kwargs) if kwargs else webdriver.Chrome()
        if self.tools:
            driver.set_window_size(height=self.tools[0].display_height, width=self.tools[0].display_width)

        if self.web_url:
            driver.get(self.web_url)
            time.sleep(3)

        res, _, usage = self._run()
        self._logger.log(message=f"Initial response: {res}", color="blue", level="info")
        self._usage.aggregate(metrics=usage)
        actions = [v for k, v in res.items() if k =="action"] if res else []
        action = actions[0] if actions else None
        action_chains = ActionChains(driver=driver)
        action_builder = ActionBuilder(driver=driver)

        if action:
            while True:
                x = action.x if hasattr(action, 'x') else 0
                y = action.y if hasattr(action, 'y') else 0
                scroll_x = action.scroll_x if hasattr(action, 'scroll_x') else 0
                scroll_y = action.scroll_y if hasattr(action, 'scroll_y') else 0
                text = action.text if hasattr(action, 'text') else ''
                path = handle_directory(directory_name='_screenshots', filename=f'cua_selenium', ext='png')

                match action.type:
                    case 'click':
                        self._logger.log(message="Action: click", color="blue", level="info")
                        driver.execute_script(f'window.scrollBy({x}, {y})')
                        action_chains.move_by_offset(xoffset=x, yoffset=y)
                        action_chains.perform()

                        if hasattr(action, 'button'):
                            match action.button:
                                case 'left':
                                    action_chains.click()
                                case 'right':
                                    action_chains.context_click()
                        action_chains.perform()
                        time.sleep(1)

                    case "scroll" | "move":
                        self._logger.log(message="Action: scroll", color="blue", level="info")
                        driver.execute_script(f'window.scrollBy({scroll_x}, {scroll_y})')
                        time.sleep(1)

                    case "keypress":
                        self._logger.log(message="Action: keypress", color="blue", level="info")
                        keys = action.keys
                        if keys:
                            for k in keys:
                                match k.lower():
                                    case "enter": action_chains.key_down(Keys.ENTER).perform()
                                    case "space": action_chains.key_down(Keys.SPACE).perform()
                                    case "select_all":
                                        if platform.system() == 'Darwin':
                                            action_chains.send_keys(Keys.COMMAND + "a").perform()
                                        else:
                                            action_chains.send_keys(Keys.CONTROL + "a").perform()
                                    case _:
                                        action_chains.key_down(Keys.SHIFT).send_keys(k).key_up(Keys.SHIFT).perform()
                        time.sleep(1)

                    case "type":
                        self._logger.log(message="Action: type", color="blue", level="info")
                        action_chains.send_keys(text).perform()
                        time.sleep(1)

                    case "wait":
                        self._logger.log(message="Action: wait", color="blue", level="info")
                        action_chains.pause(3)

                    case "screenshot":
                        self._logger.log(message="Action: screenshot", color="blue", level="info")
                        driver.save_screenshot(path)
                        time.sleep(1)

                    case _:
                        self._logger.log(message=f"Unrecognized action: {action}", level="warning", color="yellow")
                        return False

                with open(path, "rb") as image_file:
                    res, usage = None, None
                    if image_file:
                        screenshot_base64 = base64.b64encode(image_file.read()).decode("utf-8")
                        res, _, usage = self._run(screenshot=screenshot_base64)
                    else:
                        res, _, usage = self._run()

                    print("res", res)

                    self._usage.aggregate(metrics=usage)
                    if not res:
                        usage.record_errors(type=ErrorType.API)
                        break

                    actions = [v for k, v in res.items() if k =="action"] if res else []
                    action = actions[0] if actions else None
                    if not action:
                        self._logger.log(message="No action found.", color="yellow", level="warning")
                        break
        else:
            self._usage.record_errors(type=ErrorType.TOOL)

        end_dt = datetime.datetime.now()
        self._usage.record_latency(start_dt=start_dt, end_dt=end_dt)
        return res


    def run(self) -> Tuple[Dict[str, Any], None, UsageMetrics]:
        """Core function to execute the tool."""

        res = None
        try:
            res = self.invoke_playwright()
        except:
            self._call_ids = []
            self._calls = dict()
            self._response_ids = []
            res = self.invoke_selenium()

        return res, None, self._usage
