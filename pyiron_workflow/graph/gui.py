import json

from typing import Optional, Union
from pyiron_workflow import Workflow  # , Node, Port, as_function_node
from pyironflow.reactflow import ReactFlowWidget
from pyiron_workflow.graph.base import Graph
import pyiron_workflow.graph.base as base
from pyironflow.treeview import TreeView
from dataclasses import dataclass
import ipywidgets as widgets
from IPython.display import display
import threading
import time


class GUILayout:
    flow_widget_width = 1200
    flow_widget_height = 800
    output_widget_width = 400


"""
Connect graph with ReactflowWidget and other GUI elements for interactive graph/workflow visualization
"""


class PyironFlowWidget:
    def __init__(
        self,
        wf: Optional[Union["Workflow", "Graph"]] = None,
        gui_layout: GUILayout = GUILayout(),
        main_widget=None,
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

        self.graph = graph
        self.main_widget = main_widget

        # if hash_nodes:
        #     self.db = hs.create_nodes_table(echo=False)
        # else:
        self.db = None
        self.hash_nodes = hash_nodes

        self.flow_widget = ReactFlowWidget(
            layout={
                "width": f"{gui_layout.flow_widget_width}px",
                "height": f"{gui_layout.flow_widget_height}px",
            }
        )
        self.flow_widget.observe(self.on_value_change, names="commands")

        layout_accordion_widgets = {
            "border": "1px solid black",
            "width": f"{gui_layout.output_widget_width}px",
            "max_height": f"{gui_layout.flow_widget_height}px",
            "overflow": "auto",
        }

        self.log_widget = widgets.Output(layout=layout_accordion_widgets)

        self.out_widget = widgets.Output(layout=layout_accordion_widgets)
        self.tree_widget = TreeView(
            log=self.log_widget, layout=layout_accordion_widgets
        )
        self.tree_widget.flow_widget = self

        self.accordion_widget = widgets.Accordion(
            layout={  # 'border': '1px solid black',
                "min_width": f"{gui_layout.output_widget_width + 50}px",
                "height": f"{gui_layout.flow_widget_height}px",
                "overflow": "auto",
            },
        )
        self.accordion_widget.children = [
            self.tree_widget.gui,
            self.out_widget,
            self.log_widget,
        ]
        self.accordion_widget.titles = ["Node Library", "Output", "Logging Info"]
        self._counter = 0
        self.update_graph_view()

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

    # handle the commands from the ReactFlowWidget
    def on_value_change(self, change):
        from IPython.display import display

        print("on_value_change: ", change["new"], change["old"], change["name"])

        self.out_widget.clear_output()

        with self.out_widget:
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
                    tab = self.main_widget.tab_widget
                    tab.set_title(tab.selected_index, self.graph.label)
                    self.update_gui()
                elif command == "saveWorkflowName":
                    print("saveWorkflowName: ", node_name)
                    self.graph.label = node_name
                    tab = self.main_widget.tab_widget
                    tab.set_title(tab.selected_index, self.graph.label)
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

                        self.accordion_widget.selected_index = 1
                        node = self.graph.nodes[node_name].node
                        code = inspect.getsource(node._func)

                        print(highlight(code, Python2Lexer(), TerminalFormatter()))

                    elif command == "run":
                        self.accordion_widget.selected_index = 1
                        self.out_widget.clear_output()
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
                        print(
                            "expanded: ",
                            node_name,
                            self.graph.nodes[node_name].expanded,
                        )
                        self.main_widget.redraw()

                    else:
                        print("node command not recognized")

                else:
                    print("graph command not recognized")

    def update_gui(self, export_data=True, sleep_time=0.2):
        opt_graph = base._optimize_graph_connections(self.graph)
        data = dict(
            #    label=graph.label,
            nodes=base._nodes_to_gui(opt_graph),
            edges=base._edges_to_gui(opt_graph),
            graph=base._graph_to_gui(opt_graph),
            date=str(self._counter),  # unique value to activate traitlets
        )
        self._counter += 1

        time.sleep(sleep_time)
        if export_data:
            self.flow_widget.mydata = json.dumps(data)

        return data

    def update_graph_view(self, sleep_time=0.2):
        if not hasattr(self, "_thread") or not self._thread.is_alive():
            self._thread = threading.Thread(
                target=self.update_gui,
                kwargs={"export_data": True, "sleep_time": sleep_time},
            )
            self._thread.start()

    def add_node(self, node_path, label):
        node = base.get_node_from_path(node_path)(label=label) 
        self.graph += node
        self.update_gui()


############################################################################################################
# pyironflow_widget
############################################################################################################


class PyironFlow:
    def __init__(
        self, wf_list=None, hash_nodes=False, gui_layout: GUILayout = GUILayout()
    ):

        if wf_list is None:
            wf_list = [Graph(label="Workflow")]

        # self._gui_layout = gui_layout
        self.hash_nodes = hash_nodes
        self.gui_layout = gui_layout

        self.wf_widgets = list()  # list of PyironFlowWidget objects
        for wf in wf_list:
            if isinstance(wf, str):
                wf = base._load_graph(wf)
            self.wf_widgets.append(
                PyironFlowWidget(wf, gui_layout=gui_layout, main_widget=self)
            )

        self.set_tab_widget()
        # Observe the selected_index attribute of the tab widget
        self.tab_widget.observe(self.on_tab_change, names="selected_index")

        selected_index = self.tab_widget.selected_index
        wf = self.wf_widgets[selected_index]
        self.accordion_widget = widgets.Output()
        with self.accordion_widget:
            display(wf.accordion_widget)

        self.gui = widgets.HBox(
            [
                self.accordion_widget,
                self.tab_widget,
            ],
            layout={"border": "1px solid black"},
        )

    def set_tab_widget(self):
        self.tab_widget = widgets.Tab()
        self.tab_widget.children = [
            widget.flow_widget for widget in self.wf_widgets
        ] + [widgets.Output()]

        self.tab_widget.titles = [wf.graph.label for wf in self.wf_widgets] + ["+"]
        return self.tab_widget

    def delete_tab(self, index):
        print(f"Deleting tab {index}")
        self.tab_widget.children = (
            self.tab_widget.children[:index] + self.tab_widget.children[index + 2 :]
        )
        self.tab_widget.titles = (
            self.tab_widget.titles[:index] + self.tab_widget.titles[index + 2 :]
        )
        self.wf_widgets = self.wf_widgets[:index] + self.wf_widgets[index + 1 :]
        self.update_accordion(0)

    def on_tab_change(self, change):
        new_index = change["new"]
        if change["name"] == "selected_index":
            if self.tab_widget.titles[new_index] == "+":
                # Create a new tab
                new_index = len(self.tab_widget.children) - 1
                new_wf = Graph(label="Workflow_" + str(new_index))
                print(
                    f"Creating new tab for {new_wf.label}",
                    len(self.wf_widgets),
                    new_index,
                )
                self.wf_widgets.append(
                    PyironFlowWidget(
                        new_wf, gui_layout=self.gui_layout, main_widget=self
                    )
                )

                # Add the new tab to the list of children of an ipywidgets.Tab widget
                self.tab_widget.children = self.tab_widget.children[:-1] + (
                    self.wf_widgets[-1].flow_widget,
                    widgets.Output(),
                )

                for i, title in enumerate(
                    list(self.tab_widget.titles[:-2])
                    + [self.wf_widgets[-1].graph.label, "+"]
                ):
                    # print(f"Setting title {i} to {title}")
                    self.tab_widget.set_title(i, title)

            selected_index = self.tab_widget.selected_index
            wf_widget = self.wf_widgets[selected_index]
            wf_widget.update_gui()
            self.graph = wf_widget.graph  # make it accessible to the pyiron_flow object
            with self.accordion_widget:
                self.accordion_widget.clear_output()
                display(wf_widget.accordion_widget)

    def redraw(self):
        # from copy import copy

        print("redraw", self.tab_widget.selected_index)
        wf_widget = self.wf_widgets[self.tab_widget.selected_index]

        graph = base.copy_graph(wf_widget.graph)
        wf_widget.graph = Graph(label=graph.label)
        wf_widget.update_graph_view(sleep_time=0.1)
        time.sleep(0.2)
        print("redraw_reset: ", wf_widget.graph.label)
        wf_widget.graph = graph
        wf_widget.update_graph_view()

    def update_tab_children(self):
        tab_children = list(self.tab_widget.children)
        for i, wf_widget in enumerate(self.wf_widgets):
            if i == self.tab_widget.selected_index:
                print("Updating tab ", i)
                tab_children[i] = wf_widget.flow_widget
        self.tab_widget.children = tab_children
