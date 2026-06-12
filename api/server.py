from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import AsyncGenerator
from contextlib import asynccontextmanager
import json
import uuid

from config import get_settings
from graph.builder import create_graph, get_graph
from models import AgentState
from langchain_core.messages import HumanMessage


settings = get_settings()
_graph = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _graph
    _graph = get_graph()
    yield


app = FastAPI(title="Multi-Agent Research System", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=5000)
    thread_id: str | None = Field(None)
    stream: bool = Field(False)


class QueryResponse(BaseModel):
    thread_id: str
    query: str
    final_report: str | None
    research_output: str | None
    analysis_output: str | None
    confidence_score: float
    completed_agents: list[str]
    error: str | None


class StreamEvent(BaseModel):
    event: str
    node: str | None = None
    data: dict | None = None
    thread_id: str


def _create_initial_state(query: str, thread_id: str) -> AgentState:
    return AgentState(
        messages=[HumanMessage(content=query)],
        query=query,
        research_output=None,
        analysis_output=None,
        final_report=None,
        next_agent="START",
        completed_agents=[],
        confidence_score=0.0,
        error=None,
        iteration_count=0,
    )


async def _stream_events(thread_id: str, query: str) -> AsyncGenerator[str, None]:
    state = _create_initial_state(query, thread_id)
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": settings.recursion_limit}
    
    try:
        async for event in _graph.astream_events(state, config=config, version="v2"):
            if event["event"] == "on_chain_start" and event["name"] in ["supervisor", "research", "analyst", "writer"]:
                yield f"data: {json.dumps(StreamEvent(event='node_start', node=event['name'], thread_id=thread_id).model_dump())}\n\n"
            elif event["event"] == "on_chain_end" and event["name"] in ["supervisor", "research", "analyst", "writer"]:
                output = event["data"].get("output", {})
                yield f"data: {json.dumps(StreamEvent(event='node_end', node=event['name'], data={'next_agent': output.get('next_agent'), 'completed_agents': output.get('completed_agents'), 'confidence_score': output.get('confidence_score')}, thread_id=thread_id).model_dump())}\n\n"
            elif event["event"] == "on_tool_end":
                yield f"data: {json.dumps(StreamEvent(event='tool_end', node=event['name'], data={'output': str(event['data'].get('output', ''))[:500]}, thread_id=thread_id).model_dump())}\n\n"
        
        final_state = await _graph.aget_state(config)
        if final_state.values:
            yield f"data: {json.dumps(StreamEvent(event='final', data={'final_report': final_state.values.get('final_report'), 'research_output': final_state.values.get('research_output'), 'analysis_output': final_state.values.get('analysis_output'), 'confidence_score': final_state.values.get('confidence_score'), 'completed_agents': final_state.values.get('completed_agents'), 'error': final_state.values.get('error')}, thread_id=thread_id).model_dump())}\n\n"
    except Exception as e:
        yield f"data: {json.dumps(StreamEvent(event='error', data={'error': str(e)}, thread_id=thread_id).model_dump())}\n\n"


@app.post("/invoke", response_model=QueryResponse)
async def invoke(request: QueryRequest):
    thread_id = request.thread_id or str(uuid.uuid4())
    state = _create_initial_state(request.query, thread_id)
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": settings.recursion_limit}
    
    try:
        result = await _graph.ainvoke(state, config=config)
        return QueryResponse(thread_id=thread_id, query=request.query, final_report=result.get("final_report"), research_output=result.get("research_output"), analysis_output=result.get("analysis_output"), confidence_score=result.get("confidence_score", 0.0), completed_agents=result.get("completed_agents", []), error=result.get("error"))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/stream")
async def stream(request: QueryRequest):
    thread_id = request.thread_id or str(uuid.uuid4())
    return StreamingResponse(_stream_events(thread_id, request.query), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Thread-ID": thread_id})


@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "multi-agent-research"}


@app.get("/threads/{thread_id}/state")
async def get_thread_state(thread_id: str):
    config = {"configurable": {"thread_id": thread_id}}
    try:
        state = await _graph.aget_state(config)
        return state.values if state.values else {}
    except Exception:
        raise HTTPException(status_code=404, detail="Thread not found")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.server:app", host=settings.api_host, port=settings.api_port, workers=settings.api_workers, reload=True)
