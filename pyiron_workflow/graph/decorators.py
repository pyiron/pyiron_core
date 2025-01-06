from dataclasses import dataclass
from collections import OrderedDict


def as_dotdict_dataclass(*args, doc_func=None, _repr_html_=None, __add__=None, **kwargs):
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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.index_dict = self._index_dict

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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def df(self):
        import pandas as pd

        return pd.DataFrame(transpose_list_of_dicts(self))

    def _repr_html_(self):
        import pandas as pd

        return self.df._repr_html_()
