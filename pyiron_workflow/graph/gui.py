import json

from typing import Optional, Union
from pyiron_workflow import Workflow  # , Node, Port, as_function_node
from pyironflow.reactflow import ReactFlowWidget
from pyiron_workflow.graph.base import Graph
import pyiron_workflow.graph.base as base
from pyironflow.treeview import TreeView
from dataclasses import dataclass
import ipywidgets as widgets

"""
Connect graph with ReactflowWidget and other GUI elements for interactive graph/workflow visualization
"""


@dataclass
class GuiWidgets:
    output: widgets = None
    tree: widgets = None
    log: widgets = None
    flow: ReactFlowWidget = None  # ReactFlowWidget()
    accordion: widgets = None
    gui: widgets = None  # central widget containing all other widgets


class PyironFlowWidget:
    def __init__(
        self,
        wf: Optional[Union["Workflow", "Graph"]] = None,
        widgets: GuiWidgets = GuiWidgets(),
        hash_nodes=False,
    ):

        if wf is None:
            # from pyiron_workflow import Workflow
            graph = Graph("Workflow")
        elif isinstance(wf, Workflow):
            graph = base.get_graph_from_wf(wf)
        elif isinstance(wf, Graph):
            graph = base.copy_graph(wf)
        else:
            raise ValueError("wf must be a Workflow or Graph object")

        self.widgets = widgets
        if self.widgets.flow is None:
            self.widgets.flow = ReactFlowWidget()
        # self.gui = self.widgets.flow

        self.graph = graph

        # if hash_nodes:
        #     self.db = hs.create_nodes_table(echo=False)
        # else:
        self.db = None

        self.widgets.flow.observe(self.on_value_change, names="commands")
        self._counter = 0
        self._accordion_index = 1

        # self.update_gui()

    def _renew_gui(self):
        self.widgets.flow = ReactFlowWidget()
        self.widgets.flow.observe(self.on_value_change, names="commands")
        self._counter = 0

    @property
    def gui(self):
        return self.widgets.flow

    def _parse_edge_string(self, edge_str):
        """
        Convert a string representation of an edge to a GraphEdge object.

        Parameters:
        edge_str (str): The string representation of the edge in the format 'source/source_handle>target/target_handle'.

        Returns:
        GraphEdge: The GraphEdge object created from the string.
        """
        source, target = edge_str.split(">")
        source, source_handle = source.split("/")
        target, target_handle = target.split("/")
        return base.GraphEdge(
            source.strip(), target.strip(), source_handle.strip(), target_handle.strip()
        )

    def on_value_change(self, change):
        from IPython.display import display

        print("on_value_change: ", change["new"], change["old"], change["name"])

        self.widgets.output.clear_output()
        # self.graph = self.update_graph_nodes()
        # self.graph = self.update_graph_edges()

        # if "done" in change["new"]:
        #     return

        with self.widgets.output:
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                print("command: ", change["new"])
                command, node_name = change["new"].split(":", 1)
                command = command.strip()
                node_name = node_name.split("-")[0].strip()
                print("node_name: ", node_name, command)

                if command == "refreshGraphView":
                    print("refreshGraphView")
                    self.update_gui()
                elif command == "saveFlow":
                    print("saveFlow")
                    base._save_graph(self.graph, overwrite=True)
                elif command == "restoreFlow":
                    print("restoreFlow")
                    self.graph = base._load_graph(f"{self.graph.label}.json")
                    self.update_gui()
                elif command == "clearFlow":
                    print("clearFlow")
                    label = self.graph.label
                    self.graph = Graph(label=label)
                    self.widgets.gui.view_flows.set_title(
                        self.widgets.gui._tab_index, self.graph.label
                    )
                    self.update_gui()
                elif command == "saveWorkflowName":
                    print("saveWorkflowName: ", node_name)
                    self.graph.label = node_name
                    self.widgets.gui.view_flows.set_title(
                        self.widgets.gui._tab_index, self.graph.label
                    )
                    self.update_gui()  # ???
                elif command == "add_node":
                    print("add_node: ", node_name)
                    self.add_node(node_name, node_name)
                elif command == "add_edge":
                    print("add_edge: ", node_name)
                    self.graph += self._parse_edge_string(node_name)
                elif command == "delete_node":
                    print("delete_node: ", node_name)
                    self.graph = base.remove_node(self.graph, node_name)
                elif command == "delete_edge":
                    print("delete_edge: ", node_name)
                    edge = self._parse_edge_string(node_name)
                    self.graph = base.remove_edge(self.graph, edge)
                elif command == "finished":
                    print("finished")
                elif command == "change_node_value":
                    print("change_node_value: ", node_name)
                    var_dict = json.loads(node_name)
                    print("var_dict: ", var_dict)
                    self.graph = base.update_input_value(
                        self.graph,
                        node_label=var_dict["label"],
                        handle=var_dict["handle"],
                        value=var_dict["value"],
                    )

                elif node_name in self.graph.nodes.keys():
                    node = self.graph.nodes[node_name].node
                    if command == "source":
                        import inspect
                        from pygments import highlight
                        from pygments.lexers import Python2Lexer
                        from pygments.formatters import TerminalFormatter

                        self.widgets.accordion.selected_index = 1
                        node = self.graph.nodes[node_name].node
                        code = inspect.getsource(node._func)

                        print(highlight(code, Python2Lexer(), TerminalFormatter()))

                    elif command == "run":
                        self.widgets.accordion.selected_index = 1
                        self.widgets.output.clear_output()
                        if self.db is None:
                            out = base.pull_node(self.graph, node.label)
                            # out = node.pull()
                        else:
                            pass
                            # out = hs.run_node(node, self.db).outputs.to_value_dict()

                        display(out)
                    elif command == "expand":
                        print(
                            "expand: ", node_name, self.graph.nodes[node_name].expanded
                        )
                        if self.graph.nodes[node_name].expanded:
                            self.graph = base._mark_node_as_collapsed(
                                self.graph, node_name
                            )
                            print("collapsed")
                        else:
                            self.graph = base._mark_node_as_expanded(
                                self.graph, node_name
                            )
                            print("expanded")
                        self.update_gui()
                    else:
                        print("node command not recognized")

                else:
                    print("graph command not recognized")

    def update_gui(self):
        import time

        print("update_gui")

        opt_graph = base._optimize_graph_connections(self.graph)
        data = dict(
            #    label=graph.label,
            nodes=base._nodes_to_gui(opt_graph),
            edges=base._edges_to_gui(opt_graph),
            graph=base._graph_to_gui(opt_graph),
            date=str(self._counter),  # unique value to activate traitlets
        )
        self._counter += 1

        self.widgets.flow.mydata = json.dumps(data)

    def add_node(self, node_path, label):
        # self.update_graph_nodes()
        self.graph += base.GraphNode(
            id=label, import_path=node_path, widget_type="customNode", label=label
        )
        self.update_gui()


############################################################################################################
# pyironflow_widget
############################################################################################################


class GUILayout:
    flow_widget_width = 1200
    flow_widget_height = 800
    output_widget_width = 400


class PyironFlow:
    def __init__(
        self, wf_list=None, hash_nodes=False, gui_layout: GUILayout = GUILayout()
    ):

        if wf_list is None:
            wf_list = [Graph(label="Workflow")]

        self._gui_layout = gui_layout
        self.accordion = widgets.Accordion(
            layout={  # 'border': '1px solid black',
                "min_width": f"{gui_layout.output_widget_width}px",
                "max_height": f"{self._gui_layout.flow_widget_height}px",
                "overflow": "auto",
            },
        )

        self.workflows = wf_list

        self.wf_widgets = list()
        for wf in self.workflows:
            # create widgets for each workflow
            out_log = widgets.Output(
                layout={
                    "border": "1px solid black",
                    "width": f"{gui_layout.output_widget_width}px",
                    "max_height": f"{self._gui_layout.flow_widget_height}px",
                    "overflow": "auto",
                }
            )
            out_widget = widgets.Output(
                layout={
                    "border": "1px solid black",
                    "min_width": f"{gui_layout.output_widget_width}px",
                    "overflow": "auto",
                }
            )
            tree_widget = TreeView(log=out_log)
            gui_widgets = GuiWidgets(
                output=out_widget, log=out_log, tree=tree_widget, gui=self
            )
            flow_widget = PyironFlowWidget(
                wf, widgets=gui_widgets, hash_nodes=hash_nodes
            )
            flow_widget.widgets.tree.flow_widget = flow_widget

            self.wf_widgets.append(flow_widget)

        self.view_flows = self.view_flows()
        self._tab_index = 0
        self.update_accordion(self._tab_index)

        for flow_widget in self.wf_widgets:
            flow_widget.widgets.accordion = self.accordion

        self.gui = widgets.HBox(
            [
                self.accordion,
                self.view_flows,
                # self.out_widget
            ],
            layout={"border": "1px solid black"},
        )

        # Observe the selected_index attribute of the tab widget
        self.view_flows.observe(self.on_tab_change, names="selected_index")

    def update_accordion(self, tab_index, old_index=None):
        print("update_accordion", tab_index)
        if old_index is not None:
            self.wf_widgets[old_index]._accordion_index = old_index
        self._tab_index = tab_index
        current_widgets = self.wf_widgets[tab_index].widgets
        self.wf_widgets[tab_index].update_gui()
        self.accordion.children = [
            current_widgets.tree.gui,
            current_widgets.output,
            current_widgets.log,
        ]
        self.accordion.titles = ["Node Library", "Output", "Logging Info"]
        self.accordion.selected_index = self.wf_widgets[tab_index]._accordion_index

    def get_workflow(self, tab_index=0):
        wf_widget = self.wf_widgets[tab_index]
        return wf_widget.get_workflow()

    def view_flows(self):
        tab = widgets.Tab()
        tab.children = [
            self.display_workflow(index) for index, _ in enumerate(self.workflows)
        ]
        tab.titles = [wf.label for wf in self.workflows]
        return tab

    def display_workflow(self, index: int, out_flow=None):
        print("display_workflow", index)
        w = self.wf_widgets[index]
        w.update_gui()

        if out_flow is None:
            out_flow = widgets.Output(
                layout={
                    "border": "1px solid black",
                    "width": f"{self._gui_layout.flow_widget_width}px",
                    "max_height": f"{self._gui_layout.flow_widget_height}px",
                }
            )

        with out_flow:
            from IPython.display import display

            display(w.gui)

        self._tab_index = index
        self.update_accordion(index)

        return out_flow

    def on_tab_change(self, change):
        if change["name"] == "selected_index":
            new_index = change["new"]
            old_index = change["old"]
            print(f"Tab changed to {new_index}")
            # Perform any additional actions needed when the tab changes
            self.update_accordion(new_index, old_index)
        
