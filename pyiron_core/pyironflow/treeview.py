import ast
import importlib
from dataclasses import dataclass
from pathlib import Path
from ipytree import Node, Tree

__author__ = "Joerg Neugebauer"
__version__ = "1.0"  # final fully integrated version
__status__ = "development"
__date__ = "Feb 2025"


@dataclass
class FunctionNode:
    name: str
    path: str | Path


@dataclass
class DataClassNode:
    name: str
    path: str | Path


@dataclass
class ErrorNode:
    """Represents a Python module that could not be parsed."""
    name: str         # display name without .py
    path: str | Path  # original Path to the file
    error: str        # error message


def get_rel_path_for_last_occurrence(path: Path, relpath_start: str) -> int:
    """Find the relative path from the last occurrence of relpath_start in path."""
    if relpath_start in path.parts:
        reversed_parts = path.parts[::-1]
        last_occurrence = len(path.parts) - 1 - reversed_parts.index(relpath_start)
        rel_path = Path(*path.parts[last_occurrence:])
        return rel_path.with_suffix("")
    return path


class TreeView:
    def __init__(self, root_path=None, flow_widget=None, log=None, layout=None):
        import copy
        if root_path is None:
            try:
                import pyiron_core.pyiron_nodes as pyiron_nodes
                pyiron_nodes = importlib.reload(pyiron_nodes)
            except Exception as e:
                if log:
                    log.append_stdout(f"Failed to import pyiron_nodes: {e}\n")
                raise
            root_path = Path(pyiron_nodes.__path__[0])
            self.node_path = "pyiron_core"
        elif isinstance(root_path, (str, Path)):
            root_path = Path(root_path)
            if not root_path.is_absolute():
                root_path = root_path.absolute()
            self.node_path = root_path.name
        else:
            raise TypeError("root_path must be a string or Path object.")

        self.path = copy.copy(root_path)
        self.flow_widget = flow_widget
        self.log = log

        if layout is None:
            layout = {"width": "400px"}
        self.gui = Tree(stripes=True, layout=layout)
        self.add_nodes(self.gui, parent_node=self.path)
        self._handle_click_is_last_event = True

    def handle_click(self, event):
        """Triggered when a node in the tree is clicked."""
        if not self._handle_click_is_last_event:
            self._handle_click_is_last_event = True
            return None
        self._handle_click_is_last_event = False

        selected_node = event["owner"]

        if isinstance(selected_node.path, Path) and selected_node.path.is_file():
            # Module clicked; show functions/classes
            self.add_module_nodes(selected_node, selected_node.path)
        elif isinstance(selected_node.path, ErrorNode):
            if self.log:
                self.log.append_stdout(
                    f"⚠ {selected_node.path.name} has errors:\n{selected_node.path.error}\n"
                )
        elif isinstance(selected_node.path, (FunctionNode, DataClassNode)):
            selected_node.on_click(selected_node)
        elif len(selected_node.nodes) == 0:
            self.add_nodes(selected_node, selected_node.path)

    def on_click(self, node):
        """Callback when a FunctionNode/DataClassNode is clicked."""
        if isinstance(node.path, ErrorNode):
            return
        if isinstance(node.path, (FunctionNode, DataClassNode)):
            path = get_rel_path_for_last_occurrence(node.path.path, self.node_path) / node.path.name
            path_str = str(path).replace("/", ".")
            if self.flow_widget is not None:
                self.flow_widget.add_node(path_str, node.path.name)

    def add_nodes(self, tree, parent_node):
        """Add folders and module files to the tree view."""
        for child in self.list_nodes(parent_node):
            # Display without .py extension for modules
            display_name = child.stem if (child.is_file() and child.suffix == ".py") else child.name
            node_tree = Node(display_name)

            if child.is_dir():
                node_tree.icon = "folder"
                node_tree.icon_style = "warning"
                node_tree.path = child
            elif child.is_file() and child.suffix == ".py":
                module_children = self.list_pyiron_nodes(child)
                if len(module_children) == 1 and isinstance(module_children[0], ErrorNode):
                    node_tree.icon = "exclamation-triangle"
                    node_tree.icon_style = "warning"
                    node_tree.tooltip = f"Error: {module_children[0].error}"
                    node_tree.path = module_children[0]
                else:
                    node_tree.icon = "archive"
                    node_tree.icon_style = "success"
                    node_tree.path = child
            else:
                continue

            tree.add_node(node_tree)
            node_tree.on_click = self.on_click
            node_tree.observe(self.handle_click, "selected")

    def add_module_nodes(self, module_tree_node, module_path):
        """Add functions/classes inside a specific Python module node."""
        for node in self.list_pyiron_nodes(module_path):
            fn_tree = Node(node.name)
            if isinstance(node, FunctionNode):
                fn_tree.icon = "codepen"
                fn_tree.icon_style = "danger"
            elif isinstance(node, DataClassNode):
                fn_tree.icon = "table"
                fn_tree.icon_style = "success"
            elif isinstance(node, ErrorNode):
                fn_tree.icon = "exclamation-triangle"
                fn_tree.icon_style = "warning"
                fn_tree.tooltip = f"Error: {node.error}"
            fn_tree.path = node
            module_tree_node.add_node(fn_tree)
            fn_tree.on_click = self.on_click
            fn_tree.observe(self.handle_click, "selected")

    def list_nodes(self, node: Path):
        """Return directories first, then Python files, both sorted alphabetically."""
        dirs = []
        files = []
        if node.is_dir():
            for child in node.iterdir():
                if child.is_dir() and not child.name.startswith(".") and not child.name.startswith("_"):
                    dirs.append(child)
            for child in node.glob("*.py"):
                if not child.name.startswith(".") and not child.name.startswith("_"):
                    files.append(child)
        dirs.sort(key=lambda n: n.name.lower())
        files.sort(key=lambda n: n.name.lower())
        return dirs + files

    @staticmethod
    def list_pyiron_nodes(file_name, decorators=(
            "as_function_node",
            "as_macro_node",
            "as_inp_dataclass_node",
            "as_out_dataclass_node",
        )):
        """Parse Python file and return FunctionNode/DataClassNode objects or ErrorNode if parsing fails."""
        try:
            with open(file_name, "r") as file:
                source = file.read()
            tree = ast.parse(source)
        except SyntaxError as e:
            return [ErrorNode(name=Path(file_name).stem, path=Path(file_name), error=str(e))]
        except Exception as e:
            return [ErrorNode(name=Path(file_name).stem, path=Path(file_name), error=str(e))]

        nodes = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                for decorator in node.decorator_list:
                    if (
                        isinstance(decorator, ast.Call)
                        and hasattr(decorator.func, "id")
                        and decorator.func.id in decorators
                    ):
                        node_name = node.name
                        nodes.append(
                            DataClassNode(name=node_name, path=Path(file_name))
                            if isinstance(node, ast.ClassDef)
                            else FunctionNode(name=node_name, path=Path(file_name))
                        )
                    elif hasattr(decorator, "id") and decorator.id in decorators:
                        node_name = node.name
                        nodes.append(
                            DataClassNode(name=node_name, path=Path(file_name))
                            if isinstance(node, ast.ClassDef)
                            else FunctionNode(name=node_name, path=Path(file_name))
                        )
        return nodes