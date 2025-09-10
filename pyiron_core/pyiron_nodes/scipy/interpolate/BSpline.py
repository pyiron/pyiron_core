from pyiron_core.pyiron_workflow import as_function_node


@as_function_node
def DesignMatrix(x, t, k: int = 3, extrapolate: bool = False, toarray: bool = True):
    from scipy.interpolate import BSpline

    matrix = BSpline.design_matrix(x, t, k, extrapolate)
    if toarray:
        matrix = matrix.toarray()
    return matrix

