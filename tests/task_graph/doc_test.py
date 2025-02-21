def test_create_and_activate_network():
    import versionhq as vhq

    task_graph = vhq.TaskGraph(directed=False, should_reform=True)

    task_a = vhq.Task(description="Research Topic")
    task_b = vhq.Task(description="Outline Post")
    task_c = vhq.Task(description="Write First Draft")

    node_a = task_graph.add_task(task=task_a)
    node_b = task_graph.add_task(task=task_b)
    node_c = task_graph.add_task(task=task_c)

    task_graph.add_dependency(
        source=node_a.identifier, target=node_b.identifier,
        dependency_type=vhq.DependencyType.FINISH_TO_START, weight=5, description="B depends on A"
    )
    task_graph.add_dependency(
        source=node_a.identifier, target=node_c.identifier,
        dependency_type=vhq.DependencyType.FINISH_TO_FINISH, lag=1, required=False, weight=3
    )
    critical_path, duration, paths = task_graph.find_critical_path()

    import uuid
    assert isinstance(task_graph, vhq.TaskGraph)
    assert [type(k) == uuid.uuid4 and isinstance(v, vhq.Node) and isinstance(v.task, vhq.Task) for k, v in task_graph.nodes.items()]
    assert [type(k) == uuid.uuid4 and isinstance(v, vhq.Edge) for k, v in task_graph.edges.items()]
    assert critical_path is not None and duration is not None and paths is not None

    last_task_output, outputs = task_graph.activate()

    assert last_task_output is not None and isinstance(last_task_output, vhq.TaskOutput)
    assert [k in task_graph.nodes.keys() and v and isinstance(v, vhq.TaskOutput) for k, v in outputs.items()]
    assert task_graph.concl == last_task_output and task_graph.concl_template == None
