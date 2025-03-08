from unittest.mock import patch


def test_draft():
    import versionhq as vhq
    from pydantic import BaseModel

    class Trip(BaseModel):
        name: str
        location: str
        description: str
        date: str
        cousine: str
        why_its_suitable: str


    with patch('builtins.input', return_value='Y'):
        task_graph = vhq.workflow(
            Trip,
            context="Planning a suprise day trip for my friend to celebrate her birthday. We live in CA and we like to have Korean food.",
            human=True
        )

        assert task_graph is not None
        assert [k == node.identifier and node.task and isinstance(node, vhq.Node) for k, node in task_graph.nodes.items()]
        assert [isinstance(edge.dependency_type, vhq.DependencyType) and isinstance(edge, vhq.Edge) for edge in task_graph.edges.values()]
        assert [k in task_graph.nodes.keys() and v.status == vhq.TaskStatus.NOT_STARTED for k, v in task_graph.nodes.items()]


def test_condition():
    import versionhq as vhq

    task_a = vhq.Task(description="draft a message")
    task_b = vhq.Task(description="ask human for feedback")
    task_c = vhq.Task(description="send a message")

    tg = vhq.TaskGraph(directed=True)

    node_a = tg.add_task(task=task_a)
    node_b = tg.add_task(task=task_b)
    node_c = tg.add_task(task=task_c)

    def cond_method(): return True

    def cond_method_with_args(**kwargs):
        item = kwargs.get("item", None)
        return True if item else False

    complex_condition = vhq.Condition(
        methods={"0": cond_method, "1": cond_method_with_args, "2": cond_method_with_args},
        args={"2": dict(item="Hi!"), },
        type=vhq.ConditionType.AND
    )

    tg.add_dependency(
        node_a.identifier, node_b.identifier,
        dependency_type=vhq.DependencyType.FINISH_TO_START,
        required=False,
        condition=vhq.Condition(methods={ "0": cond_method, }),
    )

    edge = [v for v in tg.edges.values()][0]
    assert isinstance(edge, vhq.Edge)
    assert edge.required == False
    assert edge.dependency_type == vhq.DependencyType.FINISH_TO_START
    assert edge.condition == vhq.Condition(methods={ "0": cond_method, })
    assert edge.dependency_met() == True


    tg.add_dependency(
        node_b.identifier, node_c.identifier,
        dependency_type=vhq.DependencyType.FINISH_TO_START,
        required=False,
        condition=vhq.Condition(
            methods={ "0": cond_method, "1": cond_method_with_args, "2": complex_condition },
            type=vhq.ConditionType.AND
        ),
    )

    edge = [v for v in tg.edges.values()][1]
    assert isinstance(edge, vhq.Edge)
    assert edge.required == False
    assert edge.dependency_type == vhq.DependencyType.FINISH_TO_START
    assert edge.condition == vhq.Condition(
        methods={ "0": cond_method, "1": cond_method_with_args, "2": complex_condition },
        type=vhq.ConditionType.AND
    )
    assert edge.dependency_met() == False
