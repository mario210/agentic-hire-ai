from langgraph.graph import StateGraph, END
from src.schema.state import AgenticHireState
from src.agents.scout import ScoutAgent
from src.agents.orchestrator import OrchestratorAgent
from src.agents.tailor import TailorAgent

MODEL_NAME = "openai/gpt-4o-mini"
EMBEDDED_LLM_MODEL_NAME = "text-embedding-3-small"

# 1. Initialize the agents
orchestrator = OrchestratorAgent(model_name=MODEL_NAME, embedded_model_name=EMBEDDED_LLM_MODEL_NAME)
scout = ScoutAgent(model_name=MODEL_NAME)
tailor = TailorAgent(model_name=MODEL_NAME)


def should_continue(state: AgenticHireState):
    """
    Conditional logic to decide whether to proceed to matching
    or stop if no jobs were found.
    """
    if not state.get("found_jobs"):
        return "end"
    return "continue"


def build_graph():
    # 2. Initialize the Graph with our State schema
    workflow = StateGraph(AgenticHireState)

    # 3. Add Nodes (The Workers)
    workflow.add_node("scout", scout)
    workflow.add_node("orchestrator", orchestrator)
    workflow.add_node("tailor", tailor)

    # 4. Set the Entry Point
    workflow.set_entry_point("scout")

    # 5. Define the Flow (Edges)
    # After scouting, we check if we found anything
    workflow.add_conditional_edges(
        "scout",
        should_continue,
        {
            "continue": "orchestrator",
            "end": END
        }
    )

    # If we have jobs, they go from Matchmaker to Tailor
    workflow.add_edge("orchestrator", "tailor")

    # After tailoring, we are done
    workflow.add_edge("tailor", END)

    # 6. Compile the graph
    return workflow.compile()

app = build_graph()