from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from config import get_settings
from models import AgentState
from agents import supervisor_node, research_node, analyst_node, writer_node


settings = get_settings()


def create_graph(checkpointer=None):
    if checkpointer is None:
        checkpointer = MemorySaver()
    
    workflow = StateGraph(AgentState)
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("research", research_node)
    workflow.add_node("analyst", analyst_node)
    workflow.add_node("writer", writer_node)
    workflow.add_edge(START, "supervisor")
    workflow.add_edge("research", "supervisor")
    workflow.add_edge("analyst", "supervisor")
    workflow.add_edge("writer", "supervisor")
    
    app = workflow.compile(checkpointer=checkpointer)
    return app


def get_graph():
    return create_graph()


graph = get_graph()
