from typing import Literal

VIRTUAL = "va_"
VINPUT = f"{VIRTUAL}i_"
VOUTPUT = f"{VIRTUAL}o_"

DELIM = "__"


def _to_virtual_label(prefix: Literal[VINPUT, VOUTPUT], root: str, *paths: str) -> str:
    label = f"{prefix}{root}"
    for path in paths:
        label += f"{DELIM}{path}"
    return label


def virtual_input_label(root: str, *paths: str) -> str:
    return _to_virtual_label(VINPUT, root, *paths)


def virtual_output_label(root: str, *paths: str) -> str:
    return _to_virtual_label(VOUTPUT, root, *paths)


def extract_node_handle(label: str) -> tuple[str, str]:
    paths = label.split(DELIM)[1:]
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
