from typing import Any, Dict, List
from textwrap import dedent


class Prompt:
    """A class to format, store, and manage a prompt."""

    task: Any = None
    agent: Any = None
    context: str = None


    def __init__(self, task, agent):
        from versionhq.task.model import Task
        from versionhq.agent.model import Agent

        self.task = task if isinstance(task, Task) else Task(description=str(task))
        self.agent = agent if isinstance(agent, Agent) else Agent(role=str(agent))


    def _draft_output_prompt(self) -> str:
        output_prompt = ""

        if self.task.pydantic_output:
            output_prompt = f"""Your response MUST STRICTLY follow the given repsonse format:
JSON schema: {str(self.task.pydantic_output)}
"""

        elif self.task.response_fields:
            output_prompt, output_formats_to_follow = "", dict()
            response_format = str(self.task._structure_response_format(model_provider=self.agent.llm.provider))
            for item in self.task.response_fields:
                if item:
                    output_formats_to_follow[item.title] = f"<Return your answer in {item.data_type.__name__}>"

            output_prompt = f"""Your response MUST be a valid JSON string that strictly follows the response format. Use double quotes for all keys and string values. Do not use single quotes, trailing commas, or any other non-standard JSON syntax.
Response format: {response_format}
Ref. Output image: {output_formats_to_follow}
"""
        else:
            output_prompt = "You MUST Return your response as a valid JSON serializable string, enclosed in double quotes. Do not use single quotes, trailing commas, or other non-standard JSON syntax."

        return dedent(output_prompt)


    def _draft_context_prompt(self) -> str:
        """
        Create a context prompt from the given context in any format: a task object, task output object, list, dict.
        """
        from versionhq.task.model import Task, TaskOutput

        context_to_add = None
        if not self.context:
            # Logger().log(level="error", color="red", message="Missing a context to add to the prompt. We'll return ''.")
            return

        match self.context:
            case str():
                context_to_add = self.context

            case Task():
                if not self.context.output:
                    res = self.context.execute()
                    context_to_add = res._to_context_prompt()

                else:
                    context_to_add = self.context.output.raw

            case TaskOutput():
                context_to_add = self.context._to_context_prompt()


            case dict():
                context_to_add = str(self.context)

            case list():
                res = ", ".join([self._draft_context_prompt(context=item) for item in self.context])
                context_to_add = res

            case _:
                pass

        return dedent(context_to_add)


    def _draft_user_prompt(self) -> str:
        output_prompt = self._draft_output_prompt()
        task_slices = [self.task.description, output_prompt, ]

        if self.context:
            context_prompt = self._draft_context_prompt()
            task_slices.insert(len(task_slices), f"Consider the following context when responding: {context_prompt}")

        return "\n".join(task_slices)


    def _format_content_prompt(self) -> Dict[str, str]:
        """Formats and returns image_url content added to the messages."""

        from versionhq._utils import is_valid_url

        content_messages = {}

        if self.task.image:
            if is_valid_url(self.task.image):
                content_messages.update({ "type": "image_url", "image_url": self.task.image })
            else:
                content_messages.update({ "type": "image_url", "image_url": { "url":  self.task.image }})

        if self.task.file:
            if is_valid_url(self.task.file):
                content_messages.update({ "type": "image_url", "image_url": self.task.file })
            else:
                content_messages.update({ "type": "image_url", "image_url": { "url":  self.task.file }})

        if self.task.audio:
            from pathlib import Path
            import base64

            audio_bytes = Path(self.audio_file_path).read_bytes()
            encoded_data = base64.b64encode(audio_bytes).decode("utf-8")
            content_messages.update({  "type": "image_url", "image_url": "data:audio/mp3;base64,{}".format(encoded_data)})

        return content_messages

    @property
    def messages(self) -> List[Dict[str, str]]:
        user_prompt = self._draft_user_prompt()
        content_prompt = self._format_content_prompt()

        messages = []
        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": user_prompt
                    },
                    content_prompt,
                ]
            })

        if self.use_developer_prompt:
            messages.append({ "role": "developer", "content": self.backstory })

        return messages
