from time import sleep
from pyiron_workflow import Node
import pandas as pd

from dataclasses import dataclass, field
from collections import OrderedDict

from typing import Union



# TODO: Move to another module
def as_dotdict_dataclass(*args, doc_func=None, **kwargs):
    # def wf_data_class(*args, doc_func=None, keys_to_store=None, **kwargs):
    """
    Extension of the python default dataclass to include methods and functionality needed for pyiron_workflows

    :param args: pass to dataclass decorator
    :param doc_func: function from which to copy docstring
    # :param keys_to_store:
    :param kwargs: pass to dataclass decorator
    :return: dataclass like object with enhanced workflow features
    """

    def wrapper(cls):
        cls = dataclass(*args, **kwargs)(cls)

        # Add/modify a variable
        if doc_func is not None:
            cls.__doc__ = doc_func.__doc__

        # Add new methods
        def keys(self):
            return self.__dict__.keys()

        def items(self):
            return [(k, self[k]) for k in self.keys()]

        def asdict(self, remove_none=False):
            if remove_none:
                return {k: self[k] for k in self.keys() if self[k] is not None}
            else:
                return {k: self[k] for k in self.keys()}

        def __getitem__(self, key):
            return self.__dict__[key]

        def __setitem__(self, key, value):
            if key in self.keys():
                self.__dict__[key] = value

        def select(self, keys_to_store=None):
            if keys_to_store is None:
                keys_to_store = self.keys()  # cls._keys_to_store
            return {k: self[k] for k in keys_to_store}

        setattr(cls, "keys", keys)
        setattr(cls, "items", items)
        setattr(cls, "asdict", asdict)
        setattr(cls, "__getitem__", __getitem__)
        setattr(cls, "__setitem__", __setitem__)
        setattr(cls, "select", select)

        return cls

    return wrapper


@as_dotdict_dataclass()
class GraphNode:
    id: str  # unique identifier for the node (no two nodes can have the same id)
    import_path: str = None  # import path to the node
    label: str = None  # label/name of the node
    parent_id: str = None  # parent node id
    level: int = 0  # level in the graph
    node: Node = None  # node object
    graph: "Graph" = None  # graph object if node is a macro node
    node_type: str = "node"  # type of node
    widget_type: str = None  # type of widget to display the node
    expanded: bool = False  # expanded or collapsed state


@as_dotdict_dataclass()
class GraphEdge:
    source: str
    target: str
    sourceHandle: str = None
    targetHandle: str = None


@as_dotdict_dataclass()
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


@as_dotdict_dataclass()
class GuiData:
    label: str = None  # label/name of the node as shown in the gui
    source_labels: list = field(default_factory=lambda: [])
    target_labels: list = field(default_factory=lambda: [])
    import_path: str = None
    target_values: list = field(default_factory=lambda: [])
    target_types: list = field(default_factory=lambda: [])
    source_values: list = field(default_factory=lambda: [])
    source_types: list = field(default_factory=lambda: [])


@as_dotdict_dataclass()
class GuiStyle:
    backgroundColor: str = "rgba(0, 255, 0, 0.5)"  # light green
    height: int = 50
    width: int = 100
    padding: int = 5
    border: str = "1px black solid"
    borderRadius: str = "10px"


class Graph:
    """
    Provide all the functionality to create a hierarchical workflow graph. Key features include:
        - Add nodes including nested graphs and edges
        - Display important graph objects such as nodes and edges as a pandas dataframe
        - Collapse and expand nodes
        - Visualize the graph using the ReactFlowWidget
    """

    def __init__(
        self,
        id: str = "graph",
        nodes: dict = None,
        edges: dict = None,
        label: str = None,
        root_node: Node = None,
    ):
        self.id = id
        if label is None:
            label = id
        self.label = label
        self.root_node = root_node

        self.nodes = nodes if nodes is not None else OrderedDict()
        self.edges = edges if edges is not None else OrderedDict()

        # nodes and edges that are currently active and not inside collapsed nodes
        self.active_nodes = OrderedDict()
        self.active_edges = OrderedDict()

    def get_graph_from_macro(self, macro_node: Node):
        graph = Graph(label=macro_node._wf_macro.label, root_node=macro_node)

        for node_label, node in macro_node._wf_macro._nodes.items():
            graph.add_node(node)
        for edge in macro_node._wf_macro._edges:
            graph.add_edge(**GraphEdge(**edge))
        return graph

    def add_node(self, node: Union["Node", "Graph", "GraphNode"], label=None):
        label = self._get_label(node, label)
        self._check_label_exists(label)
        node.label = label

        if isinstance(node, Node):
            self._add_node_instance(node, label)
        elif isinstance(node, Graph):
            self._add_graph_instance(node, label)
        elif isinstance(node, GraphNode):
            self.nodes[label] = node
        else:
            raise TypeError(f"Unexpected node type {type(node)}")
        self._expand_node(label)  # TODO: validate that it is recursive

    def _get_label(self, node, label):
        if label is None:
            label = node.label
        return label

    def _check_label_exists(self, label):
        if label in self.nodes.keys():
            raise ValueError(f"Node with label {label} already exists")

    def _add_node_instance(self, node, label):
        self.nodes[label] = GraphNode(
            id=label,
            node=node,
            label=label,
            widget_type="customNode",
        )

    def _add_graph_instance(self, node, label):
        node.id = label
        # if node.root_node is not None:
        #     node.root_node.label = label
        self.nodes[label] = GraphNode(
            id=label,
            node=node.root_node,
            graph=node,
            label=label,
            node_type="graph",
            widget_type="customNode",
        )

    def add_edge(
        self,
        source: str = None,
        target: str = None,
        sourceHandle: str = None,
        targetHandle: str = None,
    ):

        n_edges = len(self.edges.keys())
        self.edges[n_edges] = GraphEdge(
            source=source,
            sourceHandle=sourceHandle,
            target=target,
            targetHandle=targetHandle,
        )

    def _expand_node(self, node_label: str):
        graph_node = self.nodes[node_label]
        if graph_node.node_type == "graph":
            graph_node.expanded = True
            node = graph_node.graph

            for k, v in node.nodes.items():
                v.parent_id = (
                    node.label
                )  # set parent id, make assignment more robust (easy to confuse id and label)
                v.level += 1
                self.nodes[k] = v

            n_edges = len(self.edges.keys())
            for k, v in node.edges.items():
                self.edges[n_edges] = v
                n_edges += 1

    def _mark_node_as_collapsed(self, node_label: str):
        graph_node = self.nodes[node_label]
        if graph_node.node_type == "graph":
            graph_node.expanded = False

    def _mark_node_as_expanded(self, node_label: str):
        graph_node = self.nodes[node_label]
        if graph_node.node_type == "graph":
            graph_node.expanded = True

    def _refresh_active_nodes(self):
        self.active_nodes = OrderedDict()
        # get all nodes that are not inside a collapsed node
        for k, v in self.nodes.items():
            if v.parent_id is None:
                self.active_nodes[k] = v
            else:
                parent = self.nodes[v.parent_id]
                if parent.expanded:
                    self.active_nodes[k] = v

    def _refresh_active_edges(self):
        self.active_edges = OrderedDict()
        # get all edges that are not inside a collapsed node
        for k, v in self.edges.items():
            if (
                v.source in self.active_nodes.keys()
                and v.target in self.active_nodes.keys()
            ):
                self.active_edges[k] = v

    def transpose_dict_of_dicts(self, nested_dict):
        transposed = {}
        for outer_key, inner_dict in nested_dict.items():
            for inner_key, value in inner_dict.items():
                if inner_key not in transposed:
                    transposed[inner_key] = []
                transposed[inner_key].append(value)
        return transposed

    def transpose_list_of_dicts(self, list_of_dicts):
        # Initialize an empty dictionary to hold the transposed data
        transposed_dict = {}

        # Iterate over each dictionary in the list
        for d in list_of_dicts:
            for key, value in d.items():
                if key not in transposed_dict:
                    transposed_dict[key] = []
                transposed_dict[key].append(value)

        return transposed_dict

    def _get_dict(self, obj: dict):
        transposed = self.transpose_dict_of_dicts(obj)
        return transposed

    def display_nodes(self):
        print(f"Graph: {self.label}")
        df = pd.DataFrame(self._get_dict(self.nodes))
        return df

    def display_edges(self):
        print(f"Graph: {self.label}")
        df = pd.DataFrame(self._get_dict(self.edges))
        return df

    def display_active_nodes(self):
        print(f"Graph: {self.label}")
        self._refresh_active_nodes()  # update active nodes, should be cached
        df = pd.DataFrame(self._get_dict(self.active_nodes))
        return df

    def display_active_edges(self):
        print(f"Graph: {self.label}")
        self._refresh_active_edges()  # update active edges, should be cached
        df = pd.DataFrame(self._get_dict(self.active_edges))
        return df

    @classmethod
    def gui_data(self, node: Node, key=None):
        from pyironflow.wf_extensions import NotData, _to_jsonifyable, _is_connected

        label = key  # node.label
        # The following does not work since the label change is not reflected in the edges
        # if (node.label != key) and (key is not None):
        #     label = f"{node.label}_{key}"

        if node is None:
            return GuiData(label=label)

        target_values = [
            _to_jsonifyable(v) if not isinstance(v, Node) else "NotData"
            for v in node.inputs.data["value"]
        ]
        is_connected = [_is_connected(v) for v in node.inputs.data["value"]]

        # TODO: set to None if it contains an edge (include connected parameter)
        target_types = [
            "None" if (t == "builtins.NoneType") or connected else t
            for t, connected in zip(node.inputs.data["type"], is_connected)
        ]

        return GuiData(
            label=label,
            source_labels=node.outputs.data["label"],
            target_labels=node.inputs.data["label"],
            import_path=node.function["import_path"],
            target_values=target_values,
            target_types=target_types,
            source_values=[NotData for _ in node.outputs.data["value"]],
            source_types=node.outputs.data["type"],
        )

    def _get_node_height(self, node: Node):
        if isinstance(node, Graph) or node is None:
            height = 250
        else:
            n_max_ports = max(node.n_out_labels, node.n_inp_labels)
            height = 30 + 16 * n_max_ports
        return height

    def _nodes_to_gui(self, remove_none=True):
        node_width = 200

        nodes = []
        self._refresh_active_nodes()
        for i, (k, v) in enumerate(self.active_nodes.items()):
            # print('node: ', k, v.label, v.node.label)
            node_dict = GuiNode(
                id=k,
                data=self.gui_data(v.node, key=k).asdict(remove_none=remove_none),
                position=dict(x=i * (node_width + 20), y=0),
                style=GuiStyle(
                    width=node_width, height=self._get_node_height(v.node)
                ).asdict(remove_none=remove_none),
                targetPosition="left",
                sourcePosition="right",
                type=v.widget_type,
                expanded=v.expanded,
            )
            if v.expanded:
                node_dict["type"] = "customNode"
                node_dict["data"] = GuiData(label=v.label).asdict(
                    remove_none=remove_none
                )
            if v.parent_id is not None:
                node_dict.parentId = v.parent_id
                node_dict.extent = "parent"
            if v.node_type == "graph":
                node_dict.type = "customNode"  # None
                node_dict.style["backgroundColor"] = "rgba(255, 165, 0, 0.3)"

            # if not v.expanded:  # for testing automated layout
            nodes.append(node_dict.asdict(remove_none=remove_none))

        return nodes

    def get_child_dict(self, node):
        if node["expanded"]:
            node_children = self._gui_children(node)
        targetPorts = [
            dict(id=f"{node['id']}_in_{label}", properties=dict(side="WEST"))
            for label in node["data"]["target_labels"]
        ][
            ::-1
        ]  # TODO: provide port positions x, y (this is only a quick fix)
        sourcePorts = [
            dict(id=f"{node['id']}_out_{label}", properties=dict(side="EAST"))
            for label in node["data"]["source_labels"]
        ][::-1]
        child = dict(
            id=node["id"],
            width=node["style"]["width"],
            height=node["style"]["height"],
            properties={"org.eclipse.elk.portConstraints": "FIXED_ORDER"},
            ports=[*targetPorts, *sourcePorts],
        )

        return child

    def _gui_children(self, gui_node):
        children = []
        nodes = self._nodes_to_gui(
            remove_none=False
        )  # TODO: cache it, avoid recomputing
        for node in nodes:
            node_children = []
            if node["parentId"] == gui_node["id"]:
                child = self.get_child_dict(node)

                child["parent"] = 1  # level in the graph? no documentation
                if len(node_children) > 0:
                    child["children"] = node_children
                children.append(child)

        return children

    def _graph_to_gui(self, remove_none=True):
        layoutOptions = {
            "elk.algorithm": "layered",
            "elk.direction": "RIGHT",
            "elk.layered.spacing.edgeNodeBetweenLayers": "40",
            "elk.spacing.nodeNode": "40",
            "elk.layered.nodePlacement.strategy": "SIMPLE",
            "elk.hierarchyHandling": "INCLUDE_CHILDREN",
        }
        graph = dict(id="root", layoutOptions=layoutOptions)

        nodes = self._nodes_to_gui(remove_none=remove_none)
        edges = self._edges_to_gui(remove_none=remove_none)
        children = []
        for node in nodes:
            if not "parentId" in node.keys():  # TODO: make this recursive
                child = self.get_child_dict(node)
                node_children = self._gui_children(node)
                if len(node_children) > 0:
                    child["children"] = node_children
                children.append(child)

        elk_edges = []
        for edge in edges:
            elk_edges.append(
                dict(
                    id=edge["id"],
                    source=edge["source"],
                    target=edge["target"],
                    sourcePort=f"{edge['source']}_out_{edge['sourceHandle']}",
                    targetPort=f"{edge['target']}_in_{edge['targetHandle']}",
                )
            )

        graph["children"] = children
        graph["edges"] = elk_edges
        return graph

    def display_gui_nodes(self):
        node_dict = self.transpose_list_of_dicts(self._nodes_to_gui(remove_none=False))
        return pd.DataFrame(node_dict)

    def display_gui_data(self):
        node_dict = self.transpose_list_of_dicts(self._nodes_to_gui(remove_none=False))
        data = pd.DataFrame(node_dict).data.to_list()

        return pd.DataFrame(self.transpose_list_of_dicts(data))

    def display_gui_style(self):
        node_dict = self.transpose_list_of_dicts(self._nodes_to_gui(remove_none=False))
        style = pd.DataFrame(node_dict).style.to_list()

        return pd.DataFrame(self.transpose_list_of_dicts(style))

    def _edges_to_gui(self, remove_none=True):
        edges = []
        self._refresh_active_edges()
        for i, (k, v) in enumerate(self.active_edges.items()):
            edge_dict = v.asdict(remove_none=remove_none)
            edge_dict["id"] = k
            edge_dict["style"] = {"strokeWidth": 2, "stroke": "black"}

            edges.append(edge_dict)

        return edges

    def on_value_change(self, change):
        # print("print command: ", change["new"])
        command, node_name = change["new"].split(":")
        if command == "finished":
            self._reactflow_widget_status = "done"
            # print("done")

    def _update_graph_view(self, w):
        import time
        import json

        w.observe(self.on_value_change, names="commands")
        self._reactflow_widget_status = "running"
        data = dict(
            #    label=graph.label,
            nodes=self._nodes_to_gui(),
            edges=self._edges_to_gui(),
            graph=self._graph_to_gui(),
        )
        time.sleep(0.2)

        w.mydata = json.dumps(data)
        for i in range(10):
            # print("waiting for done: ", i)
            if self._reactflow_widget_status == "done":
                break
            time.sleep(0.2)
        else:
            raise ValueError("Timeout: ReactFlowWidget did not finish")

        time.sleep(0.2)  # wait to give the gui time to finalize the graph

    def display(self):
        import threading
        from pyironflow.reactflow import ReactFlowWidget

        w = ReactFlowWidget()

        thread = threading.Thread(target=self._update_graph_view, args=(w,))
        thread.start()
        return w
