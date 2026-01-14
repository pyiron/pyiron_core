from typing import List, Literal

from pyiron_core.pyiron_workflow import as_function_node


@as_function_node
def ListElements(category: Literal["light", "alkali", "3d"] = "light") -> List[str]:
    """
    Return a list of element symbols for a given chemical‑group category.

    Parameters
    ----------
    category : Literal["light", "alkali", "3d"]
        The chemical group to return.  The supported groups are

        * ``"light"`` – H, He, Li, Be, B, C, N, O, F, Ne
        * ``"alkali"`` – Li, Na, K, Rb, Cs, Fr
        * ``"3d"`` – Sc, Ti, V, Cr, Mn, Fe, Co, Ni, Cu, Zn

    Returns
    -------
    List[str]
        List containing the element symbols of the selected group.
    """
    # initialise the result variable – required so that a value is always returned
    elements: List[str] = []

    if category == "light":
        elements = ["H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne"]
    elif category == "alkali":
        elements = ["Li", "Na", "K", "Rb", "Cs", "Fr"]
    elif category == "3d":
        elements = [
            "Sc",
            "Ti",
            "V",
            "Cr",
            "Mn",
            "Fe",
            "Co",
            "Ni",
            "Cu",
            "Zn",
        ]

    return elements
