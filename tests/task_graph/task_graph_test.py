def test_draft():
    import versionhq as vhq
    from pydantic import BaseModel

    class Test(BaseModel):
        name: str
        location: str
        description: str
        date: str
        cousine: str
        why_its_suitable: str

    task_graph = vhq.workflow(Test, context="Planning a suprise trip to my friend for her birthday.", human=True)

    assert task_graph
    assert [k == node.identifier and node.task and isinstance(node, vhq.Node) for k, node in task_graph.nodes.items()]
    assert [isinstance(edge.dependency_type, vhq.DependencyType) and isinstance(edge, vhq.Edge) for k, edge in task_graph.edges.items()]
    assert [k in task_graph.nodes.keys() and status == vhq.TaskStatus.NOT_STARTED for k, status in task_graph.status.items()]
