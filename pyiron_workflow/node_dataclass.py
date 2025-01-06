import os
import importlib
from pathlib import Path
from importlib.util import spec_from_file_location, module_from_spec

from dataclasses import dataclass, fields


# from pyironflow.reactflow import get_import_path

def get_import_path(obj):
    module = obj.__module__ if hasattr(obj, "__module__") else obj.__class__.__module__
    # name = obj.__name__ if hasattr(obj, "__name__") else obj.__class__.__name__
    name = obj.__name__ if "__name__" in dir(obj) else obj.__class__.__name__
    path = f"{module}.{name}"
    if path == "numpy.ndarray":
        path = "numpy.array"
    return path


def get_module_path(module_name='pyiron_nodes'):
    try:
        # Import the module
        mod = importlib.import_module(module_name)
    except ImportError:
        return "Module {} not found.".format(module_name)

    # If the module was successfully imported
    if mod:
        # If the module is a built-in module (like math, sys),
        # it won't have a __file__ attribute, so we handle it
        if hasattr(mod, '__file__'):
            return Path(mod.__file__).parent
        else:
            return "Module {} is a built-in module and doesn't have a file path.".format(module_name)
    else:
        return "Unable to find the path of Module: {}".format(module_name)


def as_output_node(data):
    lib_path = '/'.join(get_import_path(data).split('.')[:-1])
    import_path = lib_path.replace('/', '.')
    # print('as_output_node import path: ', lib_path)
    # print('data.__module__', data.__module__)
    dc = dataclass(data)
    # Generate code
    code = generate_code(dc(), import_path)
    # print (code)

    # Create directory for writing the (temporary) modules
    temp_dir = get_module_path() / 'tmp' / lib_path
    # temp_dir = os.path.expanduser('~/temp_pyiron_nodes/')
    # Create the directory
    os.makedirs(temp_dir, exist_ok=True)
    # Create the file path
    file_name = os.path.join(temp_dir, 'temp_file.py')

    # Write code to file
    with open(file_name, 'w') as temp_file:
        temp_file.write(code)

    # Import the function from the temp file
    spec = spec_from_file_location(lib_path, file_name)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)

    node = getattr(module, dc.__name__)
    node.dataclass = dc

    import inspect
    # print('code_1: ', inspect.getsource(node.node_function))

    # Delete the temporary file when done
    # os.remove(file_name)
    node.__module__ = data.__module__
    # print('node.__module__', data.__module__, node.__module__)
    # print('code_2: ', inspect.getsource(node.node_function))
    return node


def get_type_str(var_type):
    if hasattr(var_type, '__name__'):
        return var_type.__name__
    else:
        return str(var_type)


def generate_code(dc: dataclass, import_path):
    name = dc.__class__.__name__
    dc_fields = [f.name for f in fields(dc)]
    fields_with_type = [f"{f.name}: {get_type_str(f.type)}" for f in fields(dc)]
    fields_code = '\n    '.join([f'{dc_field} = dc.{dc_field}' for dc_field in dc_fields])
    return_code = ', '.join(dc_fields)
    return f'''
    
from dataclasses import dataclass
# the following line fails due to circular import, which is a result of the imports in .__init__.py in the 
# pyiron_nodes library
# from {import_path} import {name}
from pyiron_workflow import as_function_node

@as_function_node
def {name}(dc: dataclass):
    {'\n    '.join(fields_with_type)}
    {fields_code}
    return {return_code}
    '''
