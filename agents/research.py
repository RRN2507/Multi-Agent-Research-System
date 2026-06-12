from langchain_groq import ChatGroq
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage
from config import get_settings
from models import AgentState
from prompts.prompts import research_prompt
from tools import tavily_search_tool, vectorstore_retriever_tool
import json


settings = get_settings()


async def research_node(state: AgentState) -> dict:
    # Simple sequential approach: search, then synthesize
    query = state["query"]
    
    # Run searches
    tavily_results = await tavily_search_tool.ainvoke({"query": query, "max_results": 2, "search_depth": "basic"})
    vector_results = await vectorstore_retriever_tool.ainvoke({"query": query, "k": 2})
    
    # Truncate
    def truncate(r, max_chars=300):
        if isinstance(r, dict):
            content = r.get("content", "") or r.get("snippet", "") or str(r)
            return content[:max_chars] + "..." if len(content) > max_chars else content
        return str(r)[:max_chars]
    
    tavily_str = "\n".join([f"- {truncate(r)}" for r in (tavily_results if isinstance(tavily_results, list) else [tavily_results])])
    vector_str = "\n".join([f"- {truncate(r)}" for r in (vector_results if isinstance(vector_results, list) else [vector_results])])
    
    # Simple synthesis with minimal prompt
    llm = ChatGroq(
        model=settings.worker_model,
        api_key=settings.groq_api_key,
        temperature=0.1,
        max_tokens=1500,
    )
    
    synthesis_prompt = f"""Query: {query}

Web Results:
{tavily_str}

Local Results:
{vector_str}

Write a brief research report:
## Research Summary
[2 sentences]

## Key Findings
- Finding 1
- Finding 2
- Finding 3

## Sources
- Source 1 - relevance
- Source 2 - relevance

## Identified Gaps
- Gap 1

## Confidence: 0.8"""
    
    response = await llm.ainvoke(synthesis_prompt)
    output_text = response.content
    
    confidence = 0.8
    for line in output_text.split('\n'):
        if 'confidence' in line.lower():
            try:
                confidence = float(''.join(filter(lambda x: x.isdigit() or x == '.', line.split(':')[-1])))
            except:
                pass
    
    return {
        "research_output": output_text,
        "confidence_score": confidence,
        "completed_agents": state.get("completed_agents", []) + ["research"],
        "messages": [AIMessage(content=output_text, name="research")],
    }
