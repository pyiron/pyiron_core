from typing import Literal

VIRTUAL = "va_"
VINPUT = f"{VIRTUAL}i_"
VOUTPUT = f"{VIRTUAL}o_"

DELIM = "__"


def concatenate(*parts: str) -> str:
    return DELIM.join(parts)


def _to_virtual_label(prefix: Literal[VINPUT, VOUTPUT], root: str, *paths: str) -> str:
    return concatenate(f"{prefix}{root}", *paths)


def virtual_input_label(root: str, *paths: str) -> str:
    return _to_virtual_label(VINPUT, root, *paths)


def virtual_output_label(root: str, *paths: str) -> str:
    return _to_virtual_label(VOUTPUT, root, *paths)


def _get_paths_without_root(label: str) -> list[str]:
    return label.split(DELIM)[1:]


def extract_node_handle(label: str) -> tuple[str, str]:
    paths = _get_paths_without_root(label)
    if len(paths) != 2:
        raise ValueError(
            f"Expected find three elements when splitting on \"{DELIM}\", but this "
            f"was not the case for the provided label: {label}"
        )
    return tuple(paths)


def is_virtual_input(label: str) -> bool:
    return label.startswith(VINPUT)


def is_virtual_output(label: str) -> bool:
    return label.startswith(VOUTPUT)


def is_virtual(label: str) -> bool:
    return label.startswith(VIRTUAL)


def handle_to_port_label(handle: str) -> str:
    if is_virtual(handle):
        paths = _get_paths_without_root(handle)
        return DELIM.join(paths)
    return handle


def handle_to_parent_label(handle: str) -> str:
    if is_virtual_input(handle):
        return handle.lstrip(VINPUT).split(DELIM)[0]
    elif is_virtual_output(handle):
        return handle.lstrip(VOUTPUT).split(DELIM)[0]
    else:
        raise NotImplementedError(
            f"Method indicates a string return, but first checks to see if the handle "
            f"was virtual. \"{handle}\" wasn't."
        )
