import dataclasses
import inspect
import json
import os
import pathlib
import threading
import time
import warnings
from typing import Optional, Union

import ipywidgets as widgets
import numpy as np
import pandas as pd
import pygments
from IPython.display import display

import pyiron_core.pyiron_database.api as pyiron_database
import pyiron_core.pyironflow.api as pyironflow
from pyiron_core import not_data, paths
from pyiron_core.pyiron_workflow import simple_workflow
from pyiron_core.pyiron_workflow.graph import (
    base,
    decorators,
    edges,
    graph_json,
    group,
    labelling,
    run,
)


class GUILayout:
    flow_widget_width = 1200
    flow_widget_height = 800
    output_widget_width = 400


def create_db(
    user: str = "joerg",
    password: str = "none",
    host: str = "localhost",
    port: int = 5432,
    database: str = "none",
):
    import pyiron_core.pyiron_database.api as pyiron_database

    if database == "none":
        database = user

    connection_str = f"postgresql://{user}:{password}@{host}:{port}/{database}"

    db = pyiron_database.PostgreSQLInstanceDatabase(connection_str)
    db.init()

    return db


"""
Connect graph with ReactflowWidget and other GUI elements for interactive graph/workflow visualization
"""


### ADDED FOR CUSTOM FEATURES
def rename_node(graph: base.Graph, old_label: str, new_label: str):
    """Rename node in graph and update edges that reference it"""
    new_graph = base.copy_graph(graph)
    if old_label not in new_graph.nodes:
        return new_graph
    node_obj = new_graph.nodes.pop(old_label)
    # TODO: too many label-like attributes (reduce to a single one!)
    node_obj.label = new_label
    node_obj.id = new_label
    node_obj.node.label = new_label
    new_graph.nodes[new_label] = node_obj
    # handle macro nodes
    for node in new_graph.nodes.values():
        if node.parent_id == old_label:
            node.parent_id = new_label

    for e in new_graph.edges:
        if e.source == old_label:
            e.source = new_label
        if e.target == old_label:
            e.target = new_label
    return new_graph


### END ADD


class PyironFlowWidget:
    def __init__(
        self,
        workflow_path: str,
        wf: Optional[Union[simple_workflow.Workflow, base.Graph]] = None,
        gui_layout: GUILayout | None = None,
        main_widget=None,
        db=None,
    ):
        gui_layout = GUILayout() if gui_layout is None else gui_layout

        if wf is None:
            graph = base.Graph("Workflow")
        elif isinstance(wf, simple_workflow.Workflow):
            graph = base.get_full_graph_from_wf(wf)
        elif isinstance(wf, base.Graph):
            graph = base.copy_graph(wf)
        else:
            raise ValueError("wf must be a Workflow or Graph object")

        self.graph = graph
        self.main_widget = main_widget
        self.db = db
        self.workflow_path = workflow_path

        self.flow_widget = pyironflow.ReactFlowWidget(
            layout={
                "width": "100%",
                "height": f"{gui_layout.flow_widget_height}px",
            }
        )
        self.flow_widget.observe(self.on_value_change, names="commands")

        layout_accordion_widgets = {
            "border": "1px solid black",
            "width": "100%",
            "max_height": f"{gui_layout.flow_widget_height}px",
            "overflow": "auto",
        }

        self.log_widget = widgets.Output(layout=layout_accordion_widgets)

        self.out_widget = widgets.Output(layout=layout_accordion_widgets)
        self.tree_widget = pyironflow.TreeView(
            log=self.log_widget, layout=layout_accordion_widgets
        )
        self.tree_widget.flow_widget = self

        self.accordion_widget = widgets.Accordion(
            layout={
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
        self._selected_nodes = None

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
        return edges.GraphEdge(
            source.strip(), target.strip(), source_handle.strip(), target_handle.strip()
        )

    # handle the commands from the ReactFlowWidget
    def on_value_change(self, change):
        print("on_value_change: ", change["new"], change["old"], change["name"])

        import logging

        out = None
        with self.log_widget:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                # Suppress DEBUG messages from pyiron_log
                logging.getLogger("pyiron_log").setLevel(logging.WARNING)

                command, node_name = change["new"].split(":", 1)
                command = command.strip()
                node_name = node_name.rsplit("-", 1)[0].strip()

                ### ADDED FOR CUSTOM FEATURES
                if command == "renameNode":
                    payload = json.loads(node_name)
                    self.graph = rename_node(
                        self.graph, payload["oldLabel"], payload["newLabel"]
                    )
                    self.update_gui()
                ### END ADD

                elif command == "refreshGraphView":
                    print("refreshGraphView")
                    self.update_gui()
                elif command == "saveFlow":
                    print("saveFlow")
                    graph_json._save_graph(
                        self.graph, overwrite=True, workflow_dir=self.workflow_path
                    )
                elif command == "restoreFlow":
                    print("restoreFlow")
                    self.graph = graph_json._load_graph(
                        f"{self.graph.label}.json", workflow_dir=self.workflow_path
                    )
                    self.update_gui()
                elif command == "clearFlow":
                    print("clearFlow")
                    label = self.graph.label
                    self.graph = base.Graph(label=label)
                    tab = self.main_widget.tab_widget
                    tab.set_title(tab.selected_index, self.graph.label)
                    self.update_gui()
                elif command == "saveWorkflowName":
                    print("saveWorkflowName: ", node_name)
                    self.graph.label = node_name
                    tab = self.main_widget.tab_widget
                    tab.set_title(tab.selected_index, self.graph.label)
                    self.update_gui()  # ???
                elif command == "selected_nodes":
                    self._selected_nodes = node_name.split(",")
                elif command == "groupSelectedNodes":
                    print("group_nodes: ", self._selected_nodes)
                    self.graph = group.create_group(self.graph, self._selected_nodes)
                    self.update_gui()

                elif command == "add_node":
                    print("add_node: ", node_name)
                    self.add_node(node_name, node_name)
                elif command == "add_edge":
                    print("add_edge: ", node_name)
                    self.graph += self._parse_edge_string(node_name)
                    self.main_widget.redraw()
                elif command == "delete_node":
                    print("delete_node: ", node_name)
                    self.graph = base.remove_node(self.graph, node_name)
                    self.update_gui()
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
                        self.accordion_widget.selected_index = 1
                        if node.node_type == "graph":
                            code = base.get_code_from_graph(
                                node.graph,
                                sort_graph=True,
                            )
                        else:
                            code = inspect.getsource(node._func)

                        out = pygments.highlight(
                            code,
                            pygments.lexers.Python2Lexer(),
                            pygments.formatters.TerminalFormatter(),
                        )

                    elif command == "run":
                        self.accordion_widget.selected_index = 1
                        self.out_widget.clear_output()
                        out = run.pull_node(
                            base.get_updated_graph(self.graph), node.label, db=self.db
                        )

                    elif command in ["expand", "collapse"]:
                        if self.graph.nodes[node_name].expanded:
                            self.graph = _mark_node_as_collapsed(self.graph, node_name)
                        else:
                            self.graph = _mark_node_as_expanded(self.graph, node_name)
                        self.main_widget.redraw()

        if out is not None:
            self.out_widget.clear_output()
            with self.out_widget:
                if command == "run":
                    display(out)
                elif command == "source":
                    print(out)

    def update_gui(self, export_data=True, sleep_time=0.2):
        opt_graph = base.get_updated_graph(self.graph)
        data = {
            "nodes": _nodes_to_gui(opt_graph),
            "edges": _edges_to_gui(opt_graph),
            "graph": _graph_to_gui(opt_graph),
            "date": str(self._counter),
        }
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
        print(f"Adding node {label} from {node_path}")
        node = base.get_node_from_path(node_path)(label=label)
        self.graph += node
        if self.graph.nodes[label].expanded:
            self.graph = _mark_node_as_collapsed(self.graph, label)
        ### ADDED FOR CUSTOM FEATURES - track newly added nodes
        if not hasattr(self.graph, "_new_nodes"):
            self.graph._new_nodes = set()
        self.graph._new_nodes.add(label)
        ### END ADD
        self.update_gui()


############################################################################################################
# pyironflow_widget
############################################################################################################


class PyironFlow:
    def __init__(
        self,
        wf_list=None,
        hash_nodes=False,
        gui_layout: GUILayout | None = None,
        db: pyiron_database.PostgreSQLInstanceDatabase | None = None,
        workflow_path: str | pathlib.Path = paths.WORKFLOW_STORAGE,
    ):
        gui_layout = GUILayout() if gui_layout is None else gui_layout

        # create empty workflow directory if it does not exist
        if not os.path.exists(workflow_path):
            os.makedirs(workflow_path)

        self.workflow_path = workflow_path
        if wf_list is None:
            wf_list = [base.Graph(label="Workflow")]

        self.hash_nodes = hash_nodes
        if hash_nodes:
            self.db = create_db() if db is None else db
            print(f"Database created: {self.db}")
        else:
            self.db = None

        self.gui_layout = gui_layout

        self.wf_widgets = []  # list of PyironFlowWidget objects
        for wf in wf_list:
            if isinstance(wf, str):
                wf = graph_json._load_graph(pathlib.Path(workflow_path) / wf)
            self.wf_widgets.append(
                PyironFlowWidget(
                    wf=wf,
                    gui_layout=gui_layout,
                    main_widget=self,
                    db=self.db,
                    workflow_path=workflow_path,
                )
            )

        self.set_tab_widget()
        # Observe the selected_index attribute of the tab widget
        self.tab_widget.observe(self.on_tab_change, names="selected_index")

        selected_index = self.tab_widget.selected_index
        wf = self.wf_widgets[selected_index]
        self.accordion_widget = widgets.Output()
        with self.accordion_widget:
            display(wf.accordion_widget)

        self.h_scroll = widgets.IntSlider(
            value=10,
            min=0,
            max=gui_layout.flow_widget_width + gui_layout.output_widget_width,
            readout=False,
        )
        self.h_scroll.layout.width = (
            f"{gui_layout.flow_widget_width+ gui_layout.output_widget_width+15}px"
        )
        self.h_scroll.layout.margin = "0px 0px 0px 0px"
        self.h_scroll.layout.border = "1px solid black"
        self.h_scroll.layout.height = "5px"

        self.h_scroll.observe(self.update_width, names="value")

        self.gui = widgets.VBox(
            [
                widgets.HBox(
                    [
                        self.accordion_widget,
                        self.tab_widget,
                    ],
                    layout={"border": "1px solid black"},
                ),
                self.h_scroll,
            ]
        )
        # call h_scroll event to set the initial width
        self.h_scroll.value = gui_layout.output_widget_width

    def update_width(self, change):
        new_width = f'{change["new"]}px'
        self.accordion_widget.layout.width = new_width
        flow_width = (
            self.gui_layout.flow_widget_width
            + self.gui_layout.output_widget_width
            - change["new"]
        )
        self.tab_widget.layout.width = f"{flow_width}px"

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
                new_wf = base.Graph(label="Workflow_" + str(new_index))
                print(
                    f"Creating new tab for {new_wf.label}",
                    len(self.wf_widgets),
                    new_index,
                )
                self.wf_widgets.append(
                    PyironFlowWidget(
                        workflow_path=self.workflow_path,
                        wf=new_wf,
                        gui_layout=self.gui_layout,
                        main_widget=self,
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
                    self.tab_widget.set_title(i, title)

            selected_index = self.tab_widget.selected_index
            wf_widget = self.wf_widgets[selected_index]
            wf_widget.update_gui()
            self.graph = wf_widget.graph  # make it accessible to the pyiron_flow object
            with self.accordion_widget:
                self.accordion_widget.clear_output()
                display(wf_widget.accordion_widget)

    def redraw(self):
        print("redraw", self.tab_widget.selected_index)
        wf_widget = self.wf_widgets[self.tab_widget.selected_index]

        graph = base.copy_graph(wf_widget.graph)
        wf_widget.graph = base.Graph(label=graph.label)
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


@decorators.as_dotdict_dataclass()
class GuiNode:
    id: str  # unique identifier for the node (no two nodes can have the same id)
    data: dict = None
    position: dict = None
    style: dict = None
    targetPosition: str = "left"
    sourcePosition: str = "right"
    type: str = None
    parentId: str = None
    extent: str = None
    expanded: bool = False


@decorators.as_dotdict_dataclass()
class GuiData:
    label: str = None  # label/name of the node as shown in the gui
    source_labels: list = dataclasses.field(default_factory=lambda: [])
    target_labels: list = dataclasses.field(default_factory=lambda: [])
    import_path: str = None
    target_values: list = dataclasses.field(default_factory=lambda: [])
    target_types: list = dataclasses.field(default_factory=lambda: [])
    source_values: list = dataclasses.field(default_factory=lambda: [])
    source_types: list = dataclasses.field(default_factory=lambda: [])
    expanded: bool = False
    target_connected: list = dataclasses.field(default_factory=lambda: [])


@decorators.as_dotdict_dataclass()
class GuiStyle:
    backgroundColor: str = "rgba(0, 255, 0, 0.5)"  # light green
    height: int = 50
    width: int = 100
    padding: int = 5
    border: str = "1px black solid"
    borderRadius: str = "10px"


def _to_jsonifyable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, simple_workflow.Port):
        value = obj.value
        if isinstance(value, (str, int, float, bool)):
            return value
        else:
            return "NonPrimitive"
    elif isinstance(obj, simple_workflow.Node):
        return "NonPrimitive"
    elif isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    else:
        return "NonPrimitive"


def gui_data(
    node: simple_workflow.Node, key: str = None, expanded: bool = False
) -> GuiData:
    label = key
    if node is None:
        return GuiData(label=label)

    target_values = [
        (
            _to_jsonifyable(v)
            if not isinstance(v, simple_workflow.Node)
            else not_data.NotData
        )
        for v in node.inputs.data["value"]
    ]
    is_connected = [
        isinstance(v, (simple_workflow.Port, simple_workflow.Node))
        for v in node.inputs.data["value"]
    ]
    target_types = [
        "None" if (t == "builtins.NoneType") or connected else t
        for t, connected in zip(node.inputs.data["type"], is_connected, strict=False)
    ]

    is_connected = [
        isinstance(v, (simple_workflow.Port, simple_workflow.Node))
        for v in node.inputs.data["value"]
    ]

    status = getattr(node, "last_run_status", "NotRun")

    return GuiData(
        label=label,
        source_labels=node.outputs.data["label"] + ["self"],
        target_labels=node.inputs.data["label"],
        import_path=node.function["import_path"],
        target_values=target_values,
        target_types=target_types,
        source_values=[not_data.NotData for _ in node.outputs.data["value"]] + [status],
        source_types=node.outputs.data["type"] + ["Node"],
        expanded=expanded,
        target_connected=is_connected,  # NEW
    )


def _nodes_to_gui(graph: base.Graph, remove_none=True) -> decorators.NestedList:
    node_width = 280
    nodes = decorators.NestedList()
    active_nodes = _get_active_nodes(graph)
    max_x = max(
        (getattr(v, "position", {}).get("x", 0) for v in active_nodes.values()),
        default=0,
    )

    for i, (k, v) in enumerate(active_nodes.items()):
        pos_x = i * (node_width + 20)
        ### ADDED: place new nodes far right
        if hasattr(graph, "_new_nodes") and k in graph._new_nodes:
            pos_x = max_x + node_width + 100
        node_dict = GuiNode(
            id=k,
            data=gui_data(v.node, key=k, expanded=v.expanded).asdict(
                remove_none=remove_none
            ),
            position={"x": pos_x, "y": 0},
            style=GuiStyle(width=node_width, height=_get_node_height(v.node)).asdict(
                remove_none=remove_none
            ),
            targetPosition="left",
            sourcePosition="right",
            type=v.widget_type,
            expanded=v.expanded,
        )
        if v.expanded:
            node_dict["type"] = "customNode"
            node_dict["data"] = GuiData(label=v.label, expanded=True).asdict(
                remove_none=remove_none
            )
        if v.parent_id is not None:
            node_dict.parentId = v.parent_id
            node_dict.extent = "parent"

        if v.node_type == "graph":
            node_dict.type = "customNode"  # None
            node_dict.style["backgroundColor"] = "rgba(255, 165, 0, 0.3)"
        elif labelling.is_virtual(v.label):
            node_dict.style["border"] = "1px black dashed"
            node_dict.style["backgroundColor"] = "rgba(50, 50, 50, 0.1)"
        elif v.node.node_type == "out_dataclass_node":
            # light purple
            node_dict.style["backgroundColor"] = "rgba(200, 200, 255, 0.3)"
        elif v.node.node_type == "inp_dataclass_node":
            # light blue
            node_dict.style["backgroundColor"] = "rgba(100, 100, 255, 0.3)"

        nodes.append(node_dict.asdict(remove_none=remove_none))
    return nodes


def _get_node_height(node: simple_workflow.Node) -> int | float:
    if isinstance(node, base.Graph) or node is None:
        height = 250
    else:
        n_max_ports = max(node.n_out_labels, node.n_inp_labels)
        height = 30 + 20 * n_max_ports
    return height


def _get_child_dict(graph, node):
    if node["expanded"]:
        _gui_children(graph, node)
    targetPorts = [
        {"id": f"{node['id']}_in_{label}", "properties": {"side": "WEST"}}
        for label in node["data"]["target_labels"]
    ][
        ::-1
    ]  # TODO: provide port positions x, y (this is only a quick fix)
    sourcePorts = [
        {"id": f"{node['id']}_out_{label}", "properties": {"side": "EAST"}}
        for label in node["data"]["source_labels"]
    ][::-1]
    child = {
        "id": node["id"],
        "width": node["style"]["width"],
        "height": node["style"]["height"],
        "properties": {"org.eclipse.elk.portConstraints": "FIXED_ORDER"},
        "ports": [*targetPorts, *sourcePorts],
    }

    return child


def _gui_children(graph, gui_node):
    children = decorators.NestedList()
    nodes = _nodes_to_gui(graph, remove_none=False)  # TODO: cache it, avoid recomputing
    for node in nodes:
        node_children = []
        if node["parentId"] == gui_node["id"]:
            child = _get_child_dict(graph, node)

            child["parent"] = 1  # level in the graph? no documentation
            if len(node_children) > 0:
                child["children"] = node_children
            children.append(child)

    return children


def _graph_to_gui(graph: base.Graph, remove_none=True, optimize=True) -> dict:
    layoutOptions = {
        "elk.algorithm": "layered",
        "elk.direction": "RIGHT",
        "elk.layered.spacing.edgeNodeBetweenLayers": "40",
        "elk.spacing.nodeNode": "40",
        "elk.layered.nodePlacement.strategy": "SIMPLE",
        "elk.hierarchyHandling": "INCLUDE_CHILDREN",
    }
    graph_dict = {"id": "root", "layoutOptions": layoutOptions}

    nodes = _nodes_to_gui(graph, remove_none=remove_none)
    edges = _edges_to_gui(graph, remove_none=remove_none)
    children = []
    for node in nodes:
        if (
            "parentID" not in node.keys()
        ):  # TODO: make this recursive, does not work yet
            child = _get_child_dict(graph, node)
            node_children = _gui_children(graph, node)
            if len(node_children) > 0:
                child["children"] = node_children
            children.append(child)

    elk_edges = decorators.NestedList()
    for edge in edges:
        elk_edges.append(
            {
                "id": edge["id"],
                "source": edge["source"],
                "target": edge["target"],
                "sourcePort": f"{edge['source']}_out_{edge['sourceHandle']}",
                "targetPort": f"{edge['target']}_in_{edge['targetHandle']}",
            }
        )

    graph_dict["children"] = children
    graph_dict["edges"] = elk_edges
    return graph_dict


def display_gui_data(graph):
    data = _nodes_to_gui(graph, remove_none=False).df.data
    return pd.DataFrame(decorators.transpose_list_of_dicts(data))


def display_gui_style(graph):
    style = _nodes_to_gui(graph, remove_none=False).df["style"]
    return pd.DataFrame(decorators.transpose_list_of_dicts(style))


def _edges_to_gui(graph, remove_none=True):
    edges = decorators.NestedList()
    active_edges = _get_active_edges(graph)
    for i, edge in enumerate(active_edges):
        edge_dict = edge.asdict(remove_none=remove_none)
        edge_dict["id"] = i
        edge_dict["style"] = {"strokeWidth": 2, "stroke": "black"}

        edges.append(edge_dict)

    return edges


class GuiGraph:
    def __init__(
        self, graph: base.Graph, full_graph=False, sleep=0.5, width=800, height=600
    ):
        if full_graph:
            self.graph = graph
        else:
            self.graph = base.get_updated_graph(graph)

        self._width = width
        self._height = height

        self._reactflow_widget_status = "ina"
        self._sleep = sleep

    def on_value_change(self, change):
        command, node_name = change["new"].split(":")
        if command == "finished":
            self._reactflow_widget_status = "done"

    def _update_graph_view(self, w):
        w.observe(self.on_value_change, names="commands")
        self._reactflow_widget_status = "running"

        opt_graph = base.copy_graph(self.graph)
        data = {
            "nodes": _nodes_to_gui(opt_graph),
            "edges": _edges_to_gui(opt_graph),
            "graph": _graph_to_gui(opt_graph),
        }
        time.sleep(0.2)

        w.mydata = json.dumps(data)

        time.sleep(self._sleep)  # wait to give the gui time to finalize the graph

    def _repr_html_(self):
        """
        Display the graph using the ReactFlowWidget.

        This method initializes a ReactFlowWidget, updates the graph view in a separate thread,
        and returns the widget for display.
        """

        w = pyironflow.ReactFlowWidget(
            layout={
                "width": f"{self._width}px",
                "height": f"{self._height}px",
            }
        )

        if not hasattr(self, "_thread") or not self._thread.is_alive():
            self._thread = threading.Thread(target=self._update_graph_view, args=(w,))
            self._thread.start()
        return display(w)


def _mark_node_as_collapsed(graph, node_label: str):
    new_graph = base.copy_graph(graph)
    graph_node = new_graph.nodes[node_label]
    if graph_node.node_type == "graph":
        graph_node.expanded = False
    return new_graph


def _mark_node_as_expanded(graph, node_label: str):
    new_graph = base.copy_graph(graph)
    graph_node = new_graph.nodes[node_label]
    if graph_node.node_type == "graph":
        graph_node.expanded = True
    return new_graph


def _get_active_nodes(graph: base.Graph) -> base.Nodes:
    active_nodes = decorators.NestedDict(obj_type=base.GraphNode)
    # get all nodes that are not inside a collapsed node
    for k, v in graph.nodes.items():
        if v.parent_id is None:
            active_nodes[k] = v
        else:
            parent = graph.nodes[v.parent_id]
            if parent.expanded:
                active_nodes[k] = v
    return active_nodes


def _get_active_edges(graph: base.Graph) -> edges.Edges:
    active_edges = decorators.NestedList(obj_type=edges.GraphEdge)
    active_nodes = _get_active_nodes(graph)
    # get all edges that are not inside a collapsed node
    for edge in graph.edges:
        if edge.source in active_nodes.keys() and edge.target in active_nodes.keys():
            active_edges.append(edge)
    return active_edges
