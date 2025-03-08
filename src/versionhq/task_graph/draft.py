import sys
from typing import Type, Any
from pydantic import BaseModel
from pydantic._internal._model_construction import ModelMetaclass
from textwrap import dedent
if 'pydantic.main' not in sys.modules:
    import pydantic.main

sys.modules['pydantic.main'].ModelMetaclass = ModelMetaclass

from versionhq.agent.model import Agent
from versionhq.task.model import ResponseField
from versionhq.task_graph.model import TaskGraph, Task, DependencyType, Node
from versionhq._utils.logger import Logger


def workflow(final_output: Type[BaseModel], context: Any = None, human: bool = False, with_memory: bool = False) -> TaskGraph | None:
    """
    Generate a TaskGraph object to generate the givne final_output most resource-efficiently.
    """

    if not final_output or not isinstance(final_output, ModelMetaclass):
        Logger().log(level="error", message="Missing an expected output in Pydantic model.", color="red")
        return None

    final_output_prompt = ", ".join([k for k in final_output.model_fields.keys()])

    if not final_output_prompt:
        Logger().log(level="error", message="Expected output is in invalid format.", color="red")
        return None

    context_prompt = f'We are designing a resource-efficient workflow using graph algorithm concepts to achieve the following goal: {final_output_prompt}.'

    dep_type_prompt = ", ".join([k for k in DependencyType._member_map_.keys()])

    vhq_graph_expert = Agent(
        role="vhq-Graph Expert",
        goal="design the most resource-efficient workflow graph to achieve the given goal",
        knowledge_sources=[
            "https://en.wikipedia.org/wiki/Graph_theory",
            "https://www.geeksforgeeks.org/graph-and-its-representations/",
            ", ".join([k for k in DependencyType._member_map_.keys()]),
        ],
        llm="gemini-2.0",
        with_memory=with_memory,
        maxit=1,
        max_retry_limit=1,
    )

    task = Task(
        description=dedent(f"Design a resource-efficient workflow to achieve the following goal: {final_output_prompt}. The workflow should consist of a list of detailed tasks that represent decision making points, each with the following information:\nname: A concise name of the task\ndescription: A concise description of the task.\nconnections: A list of target tasks that this task connects to.\ndependency_types: The type of dependency between this task and each of its connected task. \noutput: key output from the task in a word.\n\nUse the following dependency types: {dep_type_prompt}.\n\nPrioritize minimizing resource consumption (computation, memory, and data transfer) when defining tasks, connections, and dependencies.  Consider how data is passed between tasks and aim to reduce unnecessary data duplication or transfer. Explain any design choices made to optimize resource usage."),
        response_fields=[
            ResponseField(title="tasks", data_type=list, items=dict, properties=[
                ResponseField(title="name", data_type=str),
                ResponseField(title="description", data_type=str),
                ResponseField(title="output", data_type=str),
                ResponseField(title="connections", data_type=list, items=str),
                ResponseField(title="dependency_types", data_type=list, items=str),
            ])
        ]
    )
    res = task.execute(agent=vhq_graph_expert, context=[context_prompt, context])

    if not res:
        return None

    task_items = res.json_dict["tasks"] if "tasks" in res.json_dict else []

    if not task_items:
        return None

    tasks, nodes = [], []

    for item in task_items:
        key = item["output"].lower().replace(" ", "_") if item["output"] else "output"
        task = Task(name=item["name"], description=item["description"], response_fields=[ResponseField(title=key, data_type=str)])
        tasks.append(task)
        nodes.append(Node(task=task))

    task_graph = TaskGraph(
        nodes={node.identifier: node for node in nodes},
        concl_format=final_output,
        concl=None,
        should_reform=True,
    )

    for res in task_items:
        if res["connections"]:
            dependency_types = [DependencyType[dt] if DependencyType[dt] else DependencyType.FINISH_TO_START for dt in res["dependency_types"]]

            for i, target_task_name in enumerate(res["connections"]):
                source = [v for v in task_graph.nodes.values() if v.task.name == res["name"]][0]
                target = [v for v in task_graph.nodes.values() if v.task.name == target_task_name][0]
                dependency_type = dependency_types[i]
                task_graph.add_dependency(
                    source=source.identifier, target=target.identifier, dependency_type=dependency_type)

    task_graph.visualize()

    if human:
        print('Proceed? Y/n:')
        x = input()

        if x.lower() == "y":
            print("ok. generating agent network")

        else:
            request = input("request?")
            print('ok. regenerating the graph based on your input: ', request)

    return task_graph
