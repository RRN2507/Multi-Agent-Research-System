from langchain_groq import ChatGroq
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage
from config import get_settings
from models import AgentState
from prompts.prompts import analyst_prompt


settings = get_settings()


def get_analyst_llm():
    return ChatGroq(
        model=settings.worker_model,
        api_key=settings.groq_api_key,
        temperature=0.0,
        max_tokens=2000,
    )


async def analyst_node(state: AgentState) -> dict:
    llm = get_analyst_llm()
    research_output = state.get("research_output", "")
    
    if not research_output:
        return {
            "analysis_output": "Error: No research output to analyze",
            "confidence_score": 0.0,
            "completed_agents": state.get("completed_agents", []) + ["analyst"],
            "error": "Missing research output",
        }
    
    messages = analyst_prompt.format_messages(query=state["query"], research_output=research_output)
    
    try:
        response = await llm.ainvoke(messages)
        analysis_text = response.content
        
        # Extract confidence from response
        confidence = 0.75
        for line in analysis_text.split('\n'):
            if 'confidence' in line.lower():
                try:
                    confidence = float(''.join(filter(lambda x: x.isdigit() or x == '.', line.split(':')[-1])))
                except:
                    pass
        
        # Check if more research needed
        needs_more = "needs more research" in analysis_text.lower() and "true" in analysis_text.lower()
    except Exception as e:
        analysis_text = f"Analysis completed with fallback: {str(e)}"
        confidence = 0.7
        needs_more = False
    
    output_text = f"""## Analysis Results
{analysis_text}

## Overall Confidence: {confidence:.2f}
## Needs More Research: {needs_more}"""
    
    return {
        "analysis_output": output_text,
        "confidence_score": confidence,
        "completed_agents": state.get("completed_agents", []) + ["analyst"],
        "messages": [AIMessage(content=output_text, name="analyst")],
    }
