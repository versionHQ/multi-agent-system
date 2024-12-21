from typing import List
from versionhq.task.model import Task, TaskOutput


def create_raw_outputs(tasks: List[Task], task_outputs: List[TaskOutput]) -> str:
    """
    Generate string context from the tasks.
    """

    context = ""
    if len(task_outputs) > 0:
        dividers = "\n\n----------\n\n"
        context = dividers.join(output.raw for output in task_outputs)

    else:
        task_outputs_from_task = [
            task.output for task in tasks if task.output is not None
        ]
        dividers = "\n\n----------\n\n"
        context = dividers.join(output.raw for output in task_outputs_from_task)

    return context
