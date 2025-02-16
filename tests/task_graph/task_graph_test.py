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

        assert task_graph
        assert [k == node.identifier and node.task and isinstance(node, vhq.Node) for k, node in task_graph.nodes.items()]
        assert [isinstance(edge.dependency_type, vhq.DependencyType) and isinstance(edge, vhq.Edge) for k, edge in task_graph.edges.items()]
        assert [k in task_graph.nodes.keys() and v.status == vhq.TaskStatus.NOT_STARTED for k, v in task_graph.nodes.items()]
