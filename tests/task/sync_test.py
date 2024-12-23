import os
from dotenv import load_dotenv

load_dotenv(override=True)
MODEL_NAME = os.environ.get("LITELLM_MODEL_NAME", "gpt-3.5-turbo")


def sync_test():
    from versionhq.task.model import ResponseField, Task
    from versionhq.agent.model import Agent
    from versionhq.task.model import Task, ResponseField

    agent_a = Agent(
        role="Demo Agent A",
        goal="""My amazing goals""",
        backstory="My amazing backstory",
        verbose=True,
        llm=MODEL_NAME,
        max_tokens=3000,
    )
    task = Task(
        description="Analyze the client's business model, target audience, and customer information and define the optimal cohort timeframe based on customer lifecycle and product usage patterns.",
        expected_output_raw=False,
        expected_output_json=True,
        output_field_list=[
            ResponseField(title="test1", type="int", required=True),
            ResponseField(title="test2", type="array", required=True),
        ],
        expected_output_pydantic=True,
        context=[],
        tools=[],
        callback=None,
    )

    print("...start to execute the task...")
    res = task.execute_sync(agent=agent_a)  # -> TaskOutput class
    print("Task is completed. ID:", res.task_id)
    print("Raw result: ", res.raw)
    print("JSON: ", res.json_dict)
    print("Pydantic: ", res.pydantic)
    return res


if __name__ == "__main__":
    sync_test()
