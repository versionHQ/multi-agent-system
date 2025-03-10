
from typing import Dict, List, Tuple, Any
from textwrap import dedent

from pydantic import InstanceOf

from versionhq._utils import is_valid_url


class Prompt:
    """A class to format, store, and manage a prompt."""

    task: Any = None
    agent: Any = None
    context: Any = None


    def __init__(self, task, agent, context):
        from versionhq.agent.model import Agent
        from versionhq.task.model import Task

        self.task = task if isinstance(task, Task) else Task(description=str(task))
        self.agent = agent if isinstance(agent, Agent) else Agent(role=str(agent))
        self.context = context


    def _draft_output_prompt(self) -> str:
        """Drafts prompt for output either from `pydantic_output` or `response_fields`"""

        from versionhq.llm.model import DEFAULT_MODEL_PROVIDER_NAME

        output_prompt = ""
        model_provider = self.agent.llm.provider if self.agent else DEFAULT_MODEL_PROVIDER_NAME

        if self.task.pydantic_output:
            output_prompt, output_formats_to_follow = "", dict()
            response_format = str(self.task._structure_response_format(model_provider=model_provider))
            for k, v in self.task.pydantic_output.model_fields.items():
                output_formats_to_follow[k] = f"<Return your answer in {v.annotation}>"

            output_prompt = f"""Your response MUST be a valid JSON string that strictly follows the response format. Use double quotes for all keys and string values. Do not use single quotes, trailing commas, or any other non-standard JSON syntax.
Response format: {response_format}
Ref. Output image: {output_formats_to_follow}
"""
        elif self.task.response_fields:
            output_prompt, output_formats_to_follow = "", dict()
            response_format = str(self.task._structure_response_format(model_provider=model_provider))
            for item in self.task.response_fields:
                if item:
                    output_formats_to_follow[item.title] = f"<Return your answer in {item.data_type.__name__}>"

            output_prompt = f"""Your response MUST be a valid JSON string that strictly follows the response format. Use double quotes for all keys and string values. Do not use single quotes, trailing commas, or any other non-standard JSON syntax.
Response format: {response_format}
Ref. Output image: {output_formats_to_follow}
"""
        else:
            output_prompt = "You MUST return your response as a valid JSON serializable string, enclosed in double quotes. Use double quotes for all keys and string values. Do NOT use single quotes, trailing commas, or other non-standard JSON syntax."

        return dedent(output_prompt)


    def _draft_context_prompt(self, context: Any = None) -> str:
        """
        Create a context prompt from the given context in any format: a task object, task output object, list, dict.
        """
        from versionhq.task.model import Task, TaskOutput

        context_to_add = None
        if not context:
            return context_to_add

        match context:
            case str():
                context_to_add = context

            case Task():
                if not context.output:
                    res = context.execute()
                    context_to_add = res._to_context_prompt()

                else:
                    context_to_add = context.output.raw

            case TaskOutput():
                context_to_add = context._to_context_prompt()


            case dict():
                context_to_add = str(context)

            case list():
                res = ", ".join([self._draft_context_prompt(context=item) for item in context])
                context_to_add = res

            case _:
                pass

        return dedent(context_to_add)


    def _format_content_prompt(self) -> Dict[str, str]:
        """Formats content (file, image, audio) prompt message."""

        import base64
        from pathlib import Path

        content_messages = {}

        if self.task.image:
            with open(self.task.image, "rb") as file:
                content = file.read()
                if content:
                    encoded_file = base64.b64encode(content).decode("utf-8")
                    img_url = f"data:image/jpeg;base64,{encoded_file}"
                    content_messages.update({ "type": "image_url", "image_url": { "url": img_url }})

        if self.task.file:
            if is_valid_url(self.task.file):
                content_messages.update({ "type": "image_url", "image_url": self.file })

        if self.task.audio and self.agent.llm.provider == "gemini":
            audio_bytes = Path(self.task.audio).read_bytes()
            encoded_data = base64.b64encode(audio_bytes).decode("utf-8")
            content_messages.update({ "type": "image_url", "image_url": "data:audio/mp3;base64,{}".format(encoded_data)})

        return content_messages


    def _find_rag_tools(self) -> List[InstanceOf[Any]]:
        """Find RAG tools from the agent and task object."""

        from versionhq.tool.rag_tool import RagTool

        tools = []
        if self.task.tools:
            [tools.append(item) for item in self.task.tools if isinstance(item, RagTool)]

        if self.agent.tools and self.task.can_use_agent_tools:
            [tools.append(item) for item in self.agent.tools if isinstance(item, RagTool)]

        return tools


    def draft_user_prompt(self) -> str:
        """Draft task prompts from its description and context."""

        output_prompt = self._draft_output_prompt()
        task_slices = [self.task.description, output_prompt, ]

        if self.context:
            context_prompt = self._draft_context_prompt(context=self.context)
            task_slices.insert(len(task_slices), f"Consider the following context when responding: {context_prompt}")

        return "\n".join(task_slices)


    def format_core(self, rag_tools: List[Any] = None) -> Tuple[str, str, List[Dict[str, str]]]:
        """Formats prompt messages sent to the LLM, then returns task prompt, developer prompt, and messages."""

        from versionhq.knowledge._utils import extract_knowledge_context
        from versionhq.memory.contextual_memory import ContextualMemory

        user_prompt = self.draft_user_prompt()
        rag_tools = rag_tools if rag_tools else self._find_rag_tools()

        if self.agent._knowledge:
            agent_knowledge = self.agent._knowledge.query(query=[user_prompt,], limit=5)
            if agent_knowledge:
                agent_knowledge_context = extract_knowledge_context(knowledge_snippets=agent_knowledge)
                if agent_knowledge_context:
                    user_prompt += agent_knowledge_context

        if rag_tools:
            for item in rag_tools:
                rag_tool_context = item.run(agent=self.agent, query=self.task.description)
                if rag_tool_context:
                    user_prompt += ",".join(rag_tool_context) if isinstance(rag_tool_context, list) else str(rag_tool_context)

        if self.agent.with_memory == True:
            contextual_memory = ContextualMemory(
                memory_config=self.agent.memory_config, stm=self.agent.short_term_memory, ltm=self.agent.long_term_memory, um=self.agent.user_memory
            )
            context_str = self._draft_context_prompt(context=self.context)
            query = f"{self.task.description} {context_str}".strip()
            memory = contextual_memory.build_context_for_task(query=query)
            if memory.strip() != "":
                user_prompt += memory.strip()


        ## comment out - training
        # if self.agent.networks and self.agent.networks._train:
        #     user_prompt = self.agent._training_handler(user_prompt=user_prompt)
        # else:
        #     user_prompt = self.agent._use_trained_data(user_prompt=user_prompt)


        content_prompt = self._format_content_prompt()

        messages = []
        if content_prompt:
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
        else:
            messages.append({ "role": "user", "content": user_prompt })

        if self.agent.use_developer_prompt:
            messages.append({ "role": "developer", "content": self.agent.backstory })

        return user_prompt, self.agent.backstory if self.agent.use_developer_prompt else None, messages
