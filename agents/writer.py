from langchain_groq import ChatGroq
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage
from config import get_settings
from models import AgentState, WriterOutput
from prompts.prompts import writer_prompt


settings = get_settings()


def get_writer_llm():
    return ChatGroq(
        model=settings.worker_model,
        api_key=settings.groq_api_key,
        temperature=0.3,
        max_tokens=settings.max_tokens,
    ).with_structured_output(WriterOutput)


async def writer_node(state: AgentState) -> dict:
    llm = get_writer_llm()
    research_output = state.get("research_output", "")
    analysis_output = state.get("analysis_output", "")
    
    if not research_output or not analysis_output:
        return {
            "final_report": "Error: Missing research or analysis output",
            "completed_agents": state.get("completed_agents", []) + ["writer"],
            "error": "Incomplete pipeline data",
        }
    
    messages = writer_prompt.format_messages(query=state["query"], research_output=research_output, analysis_output=analysis_output)
    
    try:
        writer_output: WriterOutput = await llm.ainvoke(messages)
        final_report = writer_output.markdown_report
    except Exception as e:
        final_report = f"""# Research Report: {state['query']}

## Executive Summary
Report generation encountered an error: {str(e)}

## Key Findings
{research_output}

## Analysis Notes
{analysis_output}

## Sources
See research output above.

## Confidence Notes
Automatic report generation failed. Manual review required."""
    
    return {
        "final_report": final_report,
        "completed_agents": state.get("completed_agents", []) + ["writer"],
        "messages": [AIMessage(content=final_report, name="writer")],
    }
