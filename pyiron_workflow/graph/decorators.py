__author__ = "Joerg Neugebauer"
__copyright__ = (
    "Copyright 2024, Max-Planck-Institut for Sustainable Materials GmbH - "
    "Computational Materials Design (CM) Department"
)
__version__ = "0.2"
__maintainer__ = ""
__email__ = ""
__status__ = "development"
__date__ = "Jan 3, 2025"

from dataclasses import dataclass
from collections import OrderedDict
import importlib


def get_import_path_from_type(obj):
    module = obj.__module__ if hasattr(obj, "__module__") else obj.__class__.__module__
    name = obj.__name__ if hasattr(obj, "__name__") else obj.__class__.__name__
    path = f"{module}.{name}"
    if path == "numpy.ndarray":
        path = "numpy.array"
    return path


def get_object_from_path(import_path, log=None):
    # Split the path into module and object part
    module_path, _, name = import_path.rpartition(".")
    # print('module_path: ', module_path)
    # Import the module
    try:
        module = importlib.import_module(module_path)
    except ModuleNotFoundError as e:
        log.append_stderr(e)
        return None
    # Get the object
    object_from_path = getattr(module, name)
    return object_from_path


def as_dotdict_dataclass(
    *args,
    doc_func=None,
    _repr_html_=None,
    __add__=None,
    __getstate__=None,
    __setstate__=None,
    **kwargs,
):
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

        if _repr_html_ is not None:
            setattr(cls, "_repr_html_", _repr_html_)

        if __add__ is not None:
            setattr(cls, "__add__", __add__)

        if __getstate__ is not None:
            setattr(cls, "__getstate__", __getstate__)

        if __setstate__ is not None:
            setattr(cls, "__setstate__", __setstate__)

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


def transpose_dict_of_dicts(nested_dict: dict) -> dict:
    transposed = {}
    for outer_key, inner_dict in nested_dict.items():
        for inner_key, value in inner_dict.items():
            if inner_key not in transposed:
                transposed[inner_key] = []
            transposed[inner_key].append(value)
    return transposed


class NestedDict(OrderedDict):
    """NestedDict is a dictionary of dictionaries where each dictionary has the same keys. A convenient way
    to display the data is in its transposed form, which represents a pandas DataFrame.
    """

    def __init__(self, obj_type=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.index_dict = self._index_dict
        self._obj_type = obj_type

    @property
    def _index_dict(self):
        return {i: k for i, k in enumerate(self.keys())}

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        self.index_dict = self._index_dict

    def __delitem__(self, key):
        super().__delitem__(key)
        self.index_dict = self._index_dict

    @property
    def df(self):
        import pandas as pd

        return pd.DataFrame(transpose_dict_of_dicts(self))

    def iloc(self, idx: int | list):
        if isinstance(idx, list):
            # make it a NestedDict
            filtered_dict = NestedDict()
            for i in idx:
                filtered_dict[self.index_dict[i]] = self[self.index_dict[i]]
            return filtered_dict
        else:
            return self[self.index_dict[idx]]

    def _repr_html_(self):
        import pandas as pd

        return self.df._repr_html_()

    def __getstate__(self):
        state = {k: v.__getstate__() for k, v in self.items()}
        if self._obj_type is not None:
            if "_obj_type" not in state:
                state.update(
                    _obj_type=get_import_path_from_type(self._obj_type),
                )
            else:
                raise ValueError("key '_obj_type' is reserved for internal use")
        return state

    def __setstate__(self, state):
        # Restore the state of the dictionary
        self.clear()
        if "_obj_type" in state:
            self._obj_type = get_object_from_path(state["_obj_type"])
            del state["_obj_type"]

            self.update({k: self._obj_type().__setstate__(v) for k, v in state.items()})
        else:
            self.update(state)
        return self    


def transpose_list_of_dicts(list_of_dicts: list) -> dict:
    # Initialize an empty dictionary to hold the transposed data
    transposed_dict = {}

    # Iterate over each dictionary in the list
    for d in list_of_dicts:
        for key, value in d.items():
            if key not in transposed_dict:
                transposed_dict[key] = []
            transposed_dict[key].append(value)

    return transposed_dict


class NestedList(list):
    """NestedList is a list of dictionaries where each dictionary has the same keys. A convinient way to display
    the data is in its transposed form, which represents a pandas DataFrame."""

    def __init__(self, obj_type=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._obj_type = obj_type

    @property
    def df(self):
        import pandas as pd

        return pd.DataFrame(transpose_list_of_dicts(self))

    def _repr_html_(self):
        import pandas as pd

        return self.df._repr_html_()

    def __getstate__(self):
        state = dict(values=[v.__getstate__() for v in self])
        if self._obj_type is not None:
            state.update(
                obj_type=get_import_path_from_type(self._obj_type),
            )
        return state

    def __setstate__(self, state):
        # Restore the state of the list
        self.clear()

        if "obj_type" in state:
            self._obj_type = get_object_from_path(state["obj_type"])
            self.extend([self._obj_type(**v) for v in state["values"]])
        else:
            self.extend(state["values"])
        return self    

    # we need to overwrite the deepcopy and copy methods to
    # prevent getting the object twice (list is double with the first part coming
    # from __setstate__)
    def __deepcopy__(self, memo):
        import copy

        # Create a new list instance
        new_list = NestedList(obj_type=self._obj_type)
        # Ensure each element is deeply copied
        for item in self:
            new_list.append(copy.deepcopy(item, memo))
        # print("deepcopy: ", memo)
        return new_list

    def __copy__(self):
        # Create a new list instance
        new_list = NestedList(obj_type=self._obj_type) 
        for item in self:
            new_list.append(item)
        return new_list
