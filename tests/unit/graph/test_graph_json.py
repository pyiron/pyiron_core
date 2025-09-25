import pathlib
import unittest

from pyiron_core.pyiron_workflow.graph import base, graph_json, group


class TestGraphJson(unittest.TestCase):
    def test_cycling_groups(self):
        g = base.Graph("has_a_group")
        g = base.add_node(g, base.identity(label="n1"))
        g = base.add_node(g, base.identity(label="n2"))
        g = base.add_node(g, base.identity(label="n3"))
        g = base.add_edge(g, "n1", "n2", "x", "x")
        g = base.add_edge(g, "n2", "n3", "x", "x")
        g = group.create_group(g, ["n1", "n2"], label="group2")

        here = pathlib.Path(__file__).parent
        fname = "with_subgraph.json"
        graph_json._save_graph(g, filename=fname, workflow_dir=str(here))

        try:
            gsl = graph_json._load_graph(fname, workflow_dir=str(here))

            self.assertSetEqual(
                set(label for label in g.nodes),
                set(label for label in gsl.nodes),
                msg="Uncollapsing node should cause graph to recover original nodes",
            )
            self.assertEqual(
                len(g.nodes),
                len(gsl.nodes),
                msg="Uncollapsed graph should have no extra nodes",
            )

            def hashable_edge(edge):
                return (edge.source, edge.target, edge.sourceHandle, edge.targetHandle)

            self.assertSetEqual(
                set(hashable_edge(edge) for edge in g.edges),
                set(hashable_edge(edge) for edge in gsl.edges),
                msg="Uncollapsing node should cause graph to recover original edges",
            )
            self.assertEqual(
                len(g.edges),
                len(gsl.edges),
                msg="Uncollapsed graph should have no extra edges",
            )
        finally:
            (here / fname).unlink(missing_ok=True)
