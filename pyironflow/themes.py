import typing

# from pyiron_workflow.nodes.function import Function
# from pyiron_workflow.nodes.macro import Macro
# from pyiron_workflow.nodes.transform import DataclassNode

if typing.TYPE_CHECKING:
    from pyiron_workflow import Node


def get_color(node: 'Node', theme: typing.Literal['light']):
    if theme == 'light':
        return light_mode(node)
    else:
        raise ValueError(f'Theme must be one of ("light",) but got {theme}')


def light_mode(node: 'Node'):
    # if isinstance(node, Function):
    if node.node_type == "function_node":
        color_light_green = "#a2ea9f"
        return color_light_green
    # elif isinstance(node, Macro):
    elif node.node_type == "macro_node":
        color_light_orange = "#eacf9f"
        return color_light_orange
    elif node.node_type == "out_dataclass_node":
    # elif isinstance(node, DataclassNode):
        color_light_purple = "#cb9fea"
        return color_light_purple
    elif node.node_type == "inp_dataclass_node":
        color_light_blue = "#9fd7ea"
        return color_light_blue    
    else:
        if hasattr(node, 'color'):
            # If the node has a color attribute, return it
            return node.color
        color_light_red = "#ea9f9f"
        # Default color for unknown node types
        print(f"Warning: Unknown node type {node.node_type}. Using default color.")
        node.color = color_light_red
        return node.color
