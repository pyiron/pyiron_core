from pyiron_workflow.graph.decorators import as_dotdict_dataclass, NestedList


def _getstate_GraphEdge(self):
    return self.asdict(remove_none=True)


def _setstate_GraphEdge(self, state):
    for k, v in state.items():
        self.__setattr__(k, v)
    return self


@as_dotdict_dataclass(
    __getstate__=_getstate_GraphEdge, __setstate__=_setstate_GraphEdge
)
class GraphEdge:
    source: str
    target: str
    sourceHandle: str = None
    targetHandle: str = None


class Edges(NestedList):
    def __init__(self, obj_type=GraphEdge):
        super().__init__(obj_type=obj_type)
