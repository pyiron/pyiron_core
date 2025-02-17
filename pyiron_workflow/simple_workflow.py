"""
workflow manager following the specs from pyiron_workflow but more oriented on functional 
programming, i.e., supports higher order functions (nodes and data as function arguments)
and implemented in a more functional (less abstract) approach.
"""

from collections import OrderedDict

import pandas as pd
import logging
from functools import wraps
import inspect
import ast
import types
from functools import partial
from typing import Any, Union
import numpy as np
import pyiron_workflow.wf_graph_tools as wf_graph_tools

PORT_LABEL = "label"
PORT_VALUE = "value"
PORT_DEFAULT = "default"
PORT_TYPE = "type"

# class NotData:
#     def __repr__(self):
#         return 'empty port'
NotData = "NotData"  # __empty"


class DotDict(OrderedDict):
    """Implementing a dot notation dictionary"""

    def __init__(self, item_type=None, *args, **kwargs):
        super().__setattr__("_item_type", item_type)
        super().__init__(*args, **kwargs)

    def __getattr__(self, attr):
        if attr in self:
            return self[attr]
        else:
            raise AttributeError(f"No such attribute: {attr}")

    def __setattr__(self, attr, value):
        self[attr] = value

    def __setitem__(self, key, value):
        if key in self.keys():
            raise TypeError(
                "'ReadOnlyDotDict' object attribute cannot be changed once set"
            )
        if self._item_type is not None:
            if not isinstance(value, self._item_type):
                raise TypeError(f"item does not match type {self._item_type}")

        super().__setitem__(key, value)

    def __delattr__(self, attr):
        if attr in self:
            del self[attr]
        else:
            raise AttributeError(f"No such attribute: {attr}")

    def add_key(self, key, value):
        super().__setattr__(key, value)
        self[key] = value

    def delete_key(self, label):
        self.__delattr__()

    def __repr__(self) -> str:
        return dict(self).__repr__()


# store and retrive types as str paths
import importlib


def get_import_path_from_type(obj):
    module = obj.__module__ if hasattr(obj, "__module__") else obj.__class__.__module__
    name = obj.__name__ if hasattr(obj, "__name__") else obj.__class__.__name__
    path = f"{module}.{name}"
    if path == "numpy.ndarray":
        path = "numpy.array"
    return path


def get_type_from_path(import_path, log=None):
    # Split the path into module and object part
    module_path, _, name = import_path.rpartition(".")
    # Import the module
    try:
        module = importlib.import_module(module_path)
    except ModuleNotFoundError as e:
        log.append_stderr(e)
        return None
    # Get the object
    object_from_path = getattr(module, name)
    return object_from_path


# extract information from function

import inspect
import ast
from typing import get_type_hints


def extract_output_parameters_from_function(func):
    # Get AST of the function's source
    func_ast = ast.parse(inspect.getsource(func))

    # Find the return node in function body
    # Returns leftmost return statement or None
    return_node = next(
        (node for node in func_ast.body[0].body if isinstance(node, ast.Return)), None
    )

    if return_node is None:
        return [], []

    # Get list of return values/expressions
    if isinstance(return_node.value, ast.Tuple):
        values = return_node.value.elts
    else:
        values = [return_node.value]

    # Generate names for all returned values
    return_names = [
        value.id if isinstance(value, ast.Name) else None for value in values
    ]

    # Get list of return type hints from the function signature
    return_annotation = inspect.signature(func).return_annotation
    if return_annotation == inspect.Signature.empty:
        return_types = [None] * len(return_names)
    elif hasattr(return_annotation, "__args__"):  # Used for typing.Tuple case
        return_types = list(return_annotation.__args__)
    else:  # Used for cases such as 'int', 'str', etc.
        return_types = [return_annotation]

    output_dict = dict()
    output_dict[PORT_LABEL] = return_names
    output_dict[PORT_TYPE] = [get_import_path_from_type(t) for t in return_types]

    return output_dict


# class NoDefaultValue:
#     def __repr__(self):
#         return 'empty'

#     def to_dict(self):
#         return {
#             '__class__': self.__class__.__name__,
#             'representation': str(self)
#         }


# Define a sentinel value. This should be a unique object that you're sure won't be used as a real default value.
# no_default = "__empty"  # NoDefaultValue()


def type_hint_to_string(type_hint: Any) -> str:
    """Convert a Python type hint to its string representation."""

    # Handling basic types
    if type_hint is int:
        return "int"
    elif type_hint is float:
        return "float"
    elif type_hint is str:
        return "str"
    elif type_hint is bool:
        return "bool"
    elif type_hint is None:
        return "None"
    elif type_hint is Node:
        return "Node"

    # Handling Optional and Union types (e.g. Optional[int], Union[int, float])
    if hasattr(type_hint, "__origin__") and (type_hint.__origin__ is Union):
        args = type_hint.__args__

        # If it contains NoneType, return the last non-NoneType element
        for arg in args[::-1]:
            if arg is not type(None):
                return type_hint_to_string(arg)

    return "NonPrimitive"


def extract_input_parameters_from_function(function: callable) -> dict:
    # Extracting function signature
    signature = inspect.signature(function)

    # Get type hints
    type_hints = get_type_hints(function)

    labels = []
    types = []
    defaults = []

    # Collecting parameter names, types, and default values
    for name, parameter in signature.parameters.items():
        type_hint = type_hints.get(
            name, None
        )  # TODO: keep here the full type info (use type_hint_to_string only when converting to gui)
        # print("type_hint: ", type_hint)
        labels.append(name)
        types.append(type_hint_to_string(type_hint))

        if parameter.default is inspect.Parameter.empty:
            defaults.append(NotData)  # no_default)
        else:
            defaults.append(parameter.default)
    # print('types: ', types)
    output_dict = dict()
    output_dict[PORT_LABEL] = labels
    output_dict[PORT_TYPE] = types
    output_dict[PORT_DEFAULT] = defaults

    return output_dict


from dataclasses import dataclass, fields, asdict


def extract_dataclass_parameters(dataclass_instance: dataclass):
    labels = [field.name for field in fields(type(dataclass_instance))]
    types = [field.type for field in fields(type(dataclass_instance))]
    defaults = [getattr(dataclass_instance, name) for name in labels]

    output_dict = dict()
    output_dict[PORT_LABEL] = labels
    output_dict[PORT_TYPE] = [type_hint_to_string(t) for t in types]
    output_dict[PORT_DEFAULT] = defaults

    return output_dict


# Node utilities


def add_field(data, **kwargs):
    for key, value in kwargs.items():
        if isinstance(value, list):
            data[key] = value
        else:
            ref_key = list(data.keys())[0]
            data[key] = [value for _ in data[ref_key]]
    return data


class Attribute:
    # TODO: add read_only option (modify via explicit update method)
    def __init__(
        self, dataset: dict, label: str, callbacks: dict = {}, read_only: bool = False
    ):
        self.__dataset = dataset
        self.__label = label
        self.__callbacks = callbacks
        self.__read_only = read_only

    @property
    def _index(self):
        return self.__dataset[PORT_LABEL].index(self.__label)

    def _to_dict(self):
        attr_dict = dict()
        for key, value in self.__dataset.items():
            attr_dict[key] = value[self._index]

        return attr_dict

    def __getattr__(self, key):
        if key.startswith("__"):
            logging.debug(key, object.__getattribute__(self, key))
            return object.__getattribute__(self, key)
        return self.__dataset[key][self._index]

    def __setattr__(self, key, value):
        logging.debug("Attribute.__setattr__", key, value)
        # if self.__read_only:
        #     raise AttributeError("This Attribute instance is read-only")
        if key.startswith("_Attribute__"):
            self.__dict__[key] = value
        else:
            self.__dataset[key][self._index] = value
            if key in self.__callbacks:
                logging.debug("callback: ", key)
                self.__callbacks[key](value)

    def __dir__(self):
        return self.__dataset.keys()

    def _repr_html_(self):
        return pd.Series(self._to_dict())

    def __repr__(self):
        return pd.Series(self._to_dict()).__repr__()
        # return self._to_dict().__repr__()


class Port(Attribute):
    def __init__(self, dataset, port_label):
        callbacks = {PORT_VALUE: self.value_changed}
        super().__init__(dataset, port_label, callbacks)

    def value_changed(self, value):
        self.ready = True
        logging.debug("value changed: value", value)

    # def __repr__(self):
    #     return f"<Port id={self.labels} value={self.value} ready={self.ready}>"


class Data:
    # TODO: make it a pure function (copy)
    def __init__(self, data, attribute=Attribute):
        super().__setattr__("data", data)
        self._attribute = attribute

    def __getattr__(self, key):
        # print('Data.__getattr__: ', key)
        if key not in self.data[PORT_LABEL]:
            raise AttributeError(f"No attribute named {key}")

        return self._attribute(self.data, key)

    def __setattr__(self, key, value):
        if key.startswith("_"):
            object.__setattr__(self, key, value)
        elif key not in self.data[PORT_LABEL]:
            raise AttributeError(f"No attribute named {key}")
        else:
            attribute = self.__getattr__(key)
            # print("Data.__setattr__: ", key, id(attribute))
            attribute.__setattr__(PORT_VALUE, value)

    def __setitem__(self, key, value):
        if key not in self.data[PORT_LABEL]:
            raise AttributeError(f"No attribute named {key}")
        else:
            attribute = self.__getattr__(key)
            attribute.__setattr__(PORT_VALUE, value)

    def __dir__(self):
        return self.data[PORT_LABEL]

    def keys(self):
        return self.__dir__()

    def __repr__(self):
        return str(self.data)

    def _repr_html_(self):
        return pd.DataFrame(self.data)._repr_html_()


# Node definition
class Node:
    def __init__(
        self,
        func=None,
        inputs=Data(
            {
                PORT_LABEL: [],
                PORT_TYPE: [],
                PORT_DEFAULT: [],
                PORT_VALUE: [],
            },
            attribute=Port,
        ),
        outputs=Data({PORT_LABEL: [], PORT_TYPE: [], PORT_VALUE: []}, attribute=Port),
        label=None,
        output_labels=None,
        node_type=None,
        orig_func=None,
    ):

        self.node_type = node_type

        self.inputs = inputs
        self.outputs = outputs

        self._func = func
        self._workflow = None

        if func is None:
            return

        if orig_func is None:
            self.function = get_function_data(func)
        else:
            self.function = get_function_data(orig_func)
            self.dataclass = orig_func  # TODO: rather specific to dataclass nodes, needs to be generalized

        # TODO: improve (add len to data)
        self.inputs.data["node"] = len(self.inputs.data[PORT_LABEL]) * [self]
        self.outputs.data["node"] = len(self.outputs.data[PORT_LABEL]) * [self]

        self.label = label
        if label is None:
            self.label = self.function.name

        if output_labels is not None:
            outs = self.outputs.data[PORT_LABEL]
            for i, (label_outs, label_out) in enumerate(zip(outs, output_labels)):
                if label_outs is None:
                    # self.outputs.data[PORT_LABEL][i] = label_out
                    outs[i] = label_out

        # validate that all output labels are given and unique
        if len(set(self.outputs.data[PORT_LABEL])) != len(
            self.outputs.data[PORT_LABEL]
        ):
            raise ValueError("Node creator: Output labels must be unique")
        if None in self.outputs.data[PORT_LABEL]:
            raise ValueError("Node creator: Output labels must be given")

        # print("node_output_labels: ", output_labels)
        # print("node_outputs_labels: ", self.outputs.data[PORT_LABEL], self.label)

    @property
    def kwargs(self):
        values = self.inputs.data[PORT_VALUE]
        labels = self.inputs.data[PORT_LABEL]
        types = self.inputs.data[PORT_TYPE]
        return {k: self._get_value(v, t) for k, v, t in zip(labels, values, types)}

    @property
    def n_out_labels(self):
        return len(self.outputs.data[PORT_LABEL])

    @property
    def n_inp_labels(self):
        return len(self.inputs.data[PORT_LABEL])

    def _get_value(self, inp_port, inp_type):
        # node_type_as_str = get_import_path_from_type(Node)
        if isinstance(inp_port, Node):
            # check whether input type is a node (provide node rather than node output value)
            if inp_type == "Node":  # node_type_as_str:
                val = inp_port
            else:
                val = inp_port.outputs.data["value"][0]
        elif isinstance(inp_port, Port):
            if (
                inp_type == "Node"
            ):  # should be used only as quick fix (node rather than port should be provided)
                val = inp_port.node
            else:
                val = inp_port.value

        else:
            val = inp_port

        return val

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        new_values = []
        labels = self.inputs.data[PORT_LABEL]
        for key in labels:
            if key in kwargs:
                new_values.append(kwargs[key])
            elif len(new_values) < len(args):
                new_values.append(args[len(new_values)])
            else:
                new_values.append(self.inputs.data[PORT_VALUE][labels.index(key)])
            # print("new_values: ", new_values)
        self.inputs.data[PORT_VALUE] = new_values
        return self.run()

    # def __getstate__(self):
    #     state = dict(
    #         label=self.label,
    #         import_path=self.function.import_path,
    #     )
    #     return state

    def _set_state(self, state):
        pass

    def run(self):
        self._validate_input()
        out = self._run()
        self._run_set_values(out)

        return out

    def _validate_input(self):
        if not all(self.inputs.data["ready"]):
            raise ValueError(
                "Input data missing", self.label, self.inputs.data["ready"]
            )

    def _run_set_values(self, out):
        for i in range(self.n_out_labels):
            if self.n_out_labels == 1:
                self.outputs.data[PORT_VALUE][i] = out
            else:
                self.outputs.data[PORT_VALUE][i] = out[i]
            self.outputs.data["ready"][i] = True

    def _run(self):
        # print("node_run: ", self.label, self.kwargs, self._func)
        return self._func(**self.kwargs)

    # TODO: quick fix only, needs serious update based on workflow graph analysis
    def pull(self):
        if self._workflow is not None:
            out = wf_graph_tools.pull_node(self._workflow, self.label)
            # self._workflow.run()
        else:
            out = self.run()
        return out
        # return self.outputs.data[PORT_VALUE]

    def _get_non_default_input(self):
        return {
            k: v
            for k, v, default in zip(
                self.inputs.data[PORT_LABEL],
                self.inputs.data[PORT_VALUE],
                self.inputs.data[PORT_DEFAULT],
            )
            if ((default == NotData) or (str(default) != str(v))) and not isinstance(v, (Node, Port))
            
        }

    def __getstate__(self):
        return dict(
            label=self.label,
            function=self.function["import_path"],
            inputs=self._get_non_default_input(),
        )

    # @classmethod
    def __setstate__(self, state):
        self = get_node_from_path(state["function"])(**state["inputs"])
        self.label = state["label"]
        return self

    @classmethod
    def from_dict(cls, node_dict):
        node = get_node_from_path(node_dict["function"])(**node_dict["inputs"])
        # print("inputs: ", node_dict["inputs"], node.inputs)
        return node


def get_node_from_path(import_path):
    # Split the path into module and object part
    module_path, _, name = import_path.rpartition(".")
    # Import the module
    try:
        module = importlib.import_module(module_path)
    except ModuleNotFoundError as e:
        return None
    # Get the object
    object_from_path = getattr(module, name)
    return object_from_path


def get_function_data(func: callable):
    function = DotDict(name=func.__name__, import_path=get_import_path_from_type(func))
    return function


def _is_equal_to_string(obj, target_string):
    if isinstance(obj, str):
        return obj == target_string
    else:
        return False  # return False if obj is not a string


def get_inputs_data(func, extract_input_parameters, *args, **kwargs):
    data = add_field(
        data=extract_input_parameters(func), ready=False, value=NotData  # PORT_VALUE
    )
    # print("default: ", data[PORT_DEFAULT])
    values_default = [
        (default if not _is_equal_to_string(default, NotData) else NotData)
        for default in data[PORT_DEFAULT]
    ]
    # print("default: ", data[PORT_DEFAULT], values_default)

    args_list = list(args)  # Convert args (a tuple) into a list

    values = [
        (kwargs[l] if l in kwargs else args_list[i] if i < len(args_list) else d)
        for i, (l, d) in enumerate(zip(data[PORT_LABEL], values_default))
    ]

    ready = [not _is_equal_to_string(value, NotData) for value in values]
    # print("values: ", values, args_list, kwargs, ready, values_default)

    data = add_field(data, value=values, ready=ready)  # PORT_VALUE
    inputs = Data(data, attribute=Port)
    return inputs


def get_outputs_data(func, extract_output_parameters):
    data = add_field(
        data=extract_output_parameters(func),
        ready=False,
        value=None,  # PORT_VALUE
    )
    outputs = Data(data, attribute=Port)
    return outputs


# The "NULL" object is used to differentiate between the decorator being called with no arguments
# or with the function to be decorated as the argument. The NULL object is unlikely to be confused
# with other parameters that users may pass to the decorator, making it a better placeholder than
# None.
NULL = object()

from functools import wraps


def make_node_decorator(
    inner_wrap_return_func, name_postfix, node_type="function_node"
):
    """Create a decorator for a function node.

    Args:
        inner_wrap_return_func (callable): A function that defines the return value of the inner_wrapper.
        name_postfix (str): Post fix for the function name.
        node_type (str, optional): Node type. Defaults to "function_node".

    Returns:
        callable: The created decorator.
    """

    def _node_decorator(*args, **kwargs):
        # label = None

        def wrapper(func=None):
            # func_print = func
            if kwargs and "labels" in kwargs:
                # print("kwargs", kwargs)
                output_labels = kwargs["labels"]
            elif args:
                # print("args", args, isinstance(args[0], str))
                if isinstance(args[0], str):
                    output_labels = list(args)
                else:
                    output_labels = None
            else:
                # print("no labels")
                output_labels = None

            if isinstance(output_labels, str):
                output_labels = [output_labels]

            # print('output_labels',  output_labels)
            # print('all: ', args, type(args), len(args), kwargs, output_labels, func)

            @wraps(func)
            def inner_wrapper(*f_args, **f_kwargs):
                import copy

                # print('inner: ', f_args, f_kwargs, inspect.getsource(func))
                if hasattr(func, "__wrapped__"):
                    print("wrapped: ", func.__wrapped__)

                cf_kwargs = copy.copy(f_kwargs)
                # print("inner_wrapper: ", inner_wrap_return_func)
                label = None
                if "label" in f_kwargs:
                    label = f_kwargs["label"]
                    del cf_kwargs["label"]
                # print('inner2: ', f_args, f_kwargs, cf_kwargs)
                return inner_wrap_return_func(
                    func, label, output_labels, node_type, *f_args, **cf_kwargs
                )

            # if hasattr(func, "__wrapped__"):
            #     print('wrapped: ', func.__wrapped__)
            inner_wrapper.__name__ += name_postfix

            return inner_wrapper

        if args and callable(args[0]):  # decorating without arguments
            return wrapper(args[0])  # passing the function to the decorator
        else:  # decorating with arguments
            return wrapper

    # print("_node_decorator", _node_decorator)
    return _node_decorator


# as_function_node decorator
def _return_as_function_node(
    func, label, output_labels, node_type, *f_args, **f_kwargs
):
    # replace this function with your custom return function
    return Node(
        func=func,
        inputs=get_inputs_data(
            func, extract_input_parameters_from_function, *f_args, **f_kwargs
        ),
        outputs=get_outputs_data(func, extract_output_parameters_from_function),
        label=label,
        output_labels=output_labels,
        node_type=node_type,
    )


as_function_node = make_node_decorator(
    _return_as_function_node, "_postfix", "function_node"
)


# as_inp_dataclass_node decorator
def _return_as_inp_dataclass_node(
    func, label, output_labels, node_type, *f_args, **f_kwargs
):

    return Node(
        func=dataclass(func),
        inputs=get_inputs_data(
            dataclass(func)(), extract_dataclass_parameters, *f_args, **f_kwargs
        ),
        outputs=Data(
            {
                "label": ["dataclass"],
                "type": ["dataclass"],
                "ready": [False],
                "value": [None],
            },
            attribute=Port,
        ),
        label=label,
        output_labels=output_labels,
        node_type=node_type,
    )


as_inp_dataclass_node = make_node_decorator(
    _return_as_inp_dataclass_node, "_postfix", "inp_dataclass_node"
)

from dataclasses import is_dataclass


# as_out_dataclass_node decorator
def _return_as_out_dataclass_node(
    func, label, output_labels, node_type, *f_args, **f_kwargs
):

    def func_dataclass(**kwargs):
        if len(kwargs.keys()) != 1:
            raise ValueError(
                "dataclass output node should contain only a single input variable",
                kwargs.keys(),
            )
        value = list(kwargs.values())[0]
        if value is None:
            raise ValueError(
                "dataclass output node should contain a non-None input variable"
            )
        return tuple(asdict(value).values())

    def find_single_element(func, *args, **kwargs):
        if args and len(args) == 1:
            return args[0]
        elif kwargs and len(kwargs) == 1:
            return next(iter(kwargs.values()))
        else:
            return dataclass(func)()

    # print("func: ", func(), is_dataclass(func()))
    return Node(
        func=func_dataclass,
        inputs=Data(
            {
                "label": ["dataclass"],
                "type": ["dataclass"],
                "ready": [True],
                "value": [find_single_element(func, *f_args, **f_kwargs)],
                "default": [dataclass(func)()],
            },
            attribute=Port,
        ),
        outputs=get_outputs_data(dataclass(func)(), extract_dataclass_parameters),
        label=label,
        output_labels=output_labels,
        node_type=node_type,
        orig_func=func,  # provide original function to get correct libpath
    )


as_out_dataclass_node = make_node_decorator(
    _return_as_out_dataclass_node, "_postfix", "out_dataclass_node"
)


# macro node decorator
def _return_as_macro_node(func, label, output_labels, node_type, *f_args, **f_kwargs):
    def run_macro(self, wf_macro):
        wf_out = wf_macro.run()  # run the workflow
        return wf_out

    # print("macro: ", inspect.getsource(func))
    node = _return_as_function_node(
        func, label, output_labels, node_type, *f_args, **f_kwargs
    )
    out = node._run()  # initialize the workflow (do not run it)
    if isinstance(out, tuple):
        out = out[0]
    wf_macro = out._workflow

    # Replace the 'run' method with a fixed argument
    node._wf_macro = wf_macro
    node._run = types.MethodType(partial(run_macro, wf_macro=wf_macro), node)
    node._orig_func = node._func  # get undecorated function

    return node


as_macro_node = make_node_decorator(_return_as_macro_node, "_postfix", "macro_node")


WORKFLOW_DIR = "../pyiron_nodes/local_workflows"


class Workflow:
    def __init__(self, label, workflow_dir=WORKFLOW_DIR):
        super().__setattr__("_nodes", OrderedDict())
        super().__setattr__("_edges", [])
        super().__setattr__("label", label)
        # super().__setattr__("workflow_dir", workflow_dir)
        super().__setattr__("_protected_names", ["add_node", "label", "run"])

    def add_node(self, label, node):
        # print(f"Adding node with value: {node} {self.child_labels}")
        if label in self.child_labels:
            # print(f"Node with label {label} already exists")
            # Split the original label and the counter if it ends with a number
            parts = label.rsplit("_", 1)
            if len(parts) > 1 and parts[-1].isdigit():
                base_label = parts[0]
                counter = int(parts[1])
            else:
                base_label = label
                counter = 1

            # Create a new unique label by incrementing the counter if needed
            label = f"{base_label}_{counter}"
            while label in self.child_labels:
                counter += 1
                label = f"{base_label}_{counter}"

            # print(f"New label: {label}")

        node.label = label
        node._workflow = self

        self._nodes[label] = node
        super().__setattr__(label, node)
        # print(f"Node added: {label} {self.child_labels} {node.label}")
        self._get_edges(node)

    def remove_node(self, label):
        if label in self.child_labels:
            del self._nodes[label]
            delattr(self, label)
            edges = [
                edge
                for edge in self._edges
                if (edge["source"] != label) & (edge["target"] != label)
            ]
            super().__setattr__("_edges", edges)
        else:
            raise ValueError(f"Node with label {label} does not exist")

    # for backward compatibility, should be removed
    def remove_child(self, label):
        self.remove_node(label)

    @property
    def child_labels(self):
        return self._nodes.keys()

    def __setattr__(self, label, value):
        # Checking if name is "node". If it is, redirect to add_node method
        if isinstance(value, Node):
            if label.startswith("_") | (label in self._protected_names):
                raise ValueError(f"Invalid name for a node label: {label}")
            if label in self.child_labels:
                raise ValueError(f"Node with label {label} already exists")
            self.add_node(label=label, node=value)
            # else:
            # If the attribute is not "node", use the regular attribute setting mechanism
            super().__setattr__(label, value)

    def _get_edges(self, node):
        values = node.inputs.data[PORT_VALUE]
        labels = node.inputs.data[PORT_LABEL]
        for l, v in zip(labels, values):
            # print('ports: ', l, type(v))
            if isinstance(v, Port):
                source = v.node.label
                sourceHandle = v.label
                target = node.label
                targetHandle = l  # v.label
            elif isinstance(v, Node):
                source_node = v
                source = source_node.label
                # print('source_node: ', source_node, source_node.outputs.data[PORT_LABEL], node.label)
                if source_node.n_out_labels == 1:
                    sourceHandle = source_node.outputs.data["label"][0]
                else:
                    raise ValueError(
                        "Node {node.label} has multiple output ports. Please specify the port to connect."
                    )

                target = node.label
                targetHandle = l
            else:
                continue

            edge = dict(
                source=source,
                sourceHandle=sourceHandle,
                target=target,
                targetHandle=targetHandle,
            )
            self._edges.append(edge)
            # print(f"{source}/{sourceHandle} -> {target}/{targetHandle}")
            logging.info(f"{source}/{sourceHandle} -> {target}/{targetHandle}")

    def _set_edges(self, value):
        super().__setattr__("_edges", value)

        # set source value to target node
        for edge in self._edges:
            source = edge["source"]
            target = edge["target"]
            sourceHandle = edge["sourceHandle"]
            targetHandle = edge["targetHandle"]
            source_node = self._nodes[source]
            target_node = self._nodes[target]

            ind_label = target_node.inputs.data[PORT_LABEL].index(targetHandle)
            num_outputs = (
                source_node.n_out_labels
            )  # len(source_node.outputs.data[PORT_LABEL])
            # print('set_edges: ', source, target, targetHandle, ind_label, num_outputs, source_node.outputs.data[PORT_LABEL])
            if num_outputs == 1:
                target_node.inputs.data[PORT_VALUE][ind_label] = source_node
            elif num_outputs > 1:
                target_node.inputs.data[PORT_VALUE][ind_label] = (
                    source_node.outputs.__getattr__(sourceHandle)
                )
            target_node.inputs.data["ready"][ind_label] = True  # TODO: make it dynamic

    def run(self, debug=False):
        return wf_graph_tools.run_wf(self, debug=debug)

    def __repr__(self):
        return "pyiron_workflow instance: \n" + wf_graph_tools.get_code_from_wf(self)

    def save(self, filename, workflow_dir=WORKFLOW_DIR, overwrite=False):
        graph = wf_graph_tools.get_graph_from_wf(self)

        return wf_graph_tools._save_graph(
            graph=graph,
            filename=filename,
            workflow_dir=workflow_dir,
            overwrite=overwrite,
        )

    @staticmethod
    def load(filename, workflow_dir=None):
        if workflow_dir is None:
            workflow_dir = WORKFLOW_DIR
        graph = wf_graph_tools._load_graph(filename=filename, workflow_dir=workflow_dir)
        wf = wf_graph_tools.get_wf_from_graph(graph)
        return wf
