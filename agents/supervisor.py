
from langchain_groq import ChatGroq
from langgraph.types import Command
from config import get_settings
from models import AgentState, SupervisorDecision
from prompts.prompts import supervisor_prompt
import json


settings = get_settings()


def get_supervisor_llm():
    return ChatGroq(
        model=settings.supervisor_model,
        api_key=settings.groq_api_key,
        temperature=0.0,
        max_tokens=500,
    )


async def supervisor_node(state: AgentState) -> Command:
    llm = get_supervisor_llm()
    
    completed = state.get("completed_agents", [])
    iteration = state.get("iteration_count", 0)
    confidence = state.get("confidence_score", 0.0)
    
    # Simple deterministic routing (more reliable than LLM for this)
    if "research" not in completed:
        next_agent = "research"
        reasoning = "Research not yet completed"
    elif "analyst" not in completed:
        next_agent = "analyst"
        reasoning = "Analysis not yet completed"
    elif "writer" not in completed:
        next_agent = "writer"
        reasoning = "Ready for report generation"
    else:
        next_agent = "END"
        reasoning = "All agents completed"
    
    # Cap iterations
    if iteration >= 10:
        next_agent = "END"
        reasoning = "Max iterations reached"
    
    goto_target = "END" if next_agent == "END" else next_agent
    
    updates = {
        "next_agent": next_agent,
        "iteration_count": iteration + 1,
        "messages": [{"role": "assistant", "content": f"Supervisor routing to: {next_agent}. Reasoning: {reasoning}", "name": "supervisor"}],
    }
    
    return Command(goto=goto_target, update=updates)
