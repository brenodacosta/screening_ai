from langgraph.graph import END, StateGraph

from .models import ConversationState, Stage
from .nodes import extract_node, render_node, route_node, terminate_node


_TERMINAL_STAGES = {Stage.QUALIFIED, Stage.DISQUALIFIED, Stage.ABANDONED}


def _branch_after_route(state: ConversationState) -> str:
    return "terminate" if state.current_stage in _TERMINAL_STAGES else "render"


def build_graph(checkpointer):
    """Compile the screening graph. `checkpointer` is a LangGraph BaseCheckpointSaver."""
    g = StateGraph(state_schema=ConversationState)

    g.add_node("extract", extract_node)
    g.add_node("route", route_node)
    g.add_node("render", render_node)
    g.add_node("terminate", terminate_node)

    g.set_entry_point("extract")
    g.add_edge("extract", "route")
    g.add_conditional_edges(
        "route",
        _branch_after_route,
        {"render": "render", "terminate": "terminate"},
    )
    g.add_edge("render", END)
    g.add_edge("terminate", END)

    return g.compile(checkpointer=checkpointer)
