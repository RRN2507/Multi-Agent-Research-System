
import asyncio
import argparse
import sys
from uuid import uuid4

from config import get_settings
from graph.builder import create_graph
from models import AgentState
from langchain_core.messages import HumanMessage
from langgraph.types import Command


settings = get_settings()


def _extract_output(output):
    """Extract dict from Command or return dict as-is."""
    if isinstance(output, Command):
        return output.update if hasattr(output, 'update') else {}
    return output if isinstance(output, dict) else {}


async def run_research(query: str, thread_id: str | None = None, verbose: bool = False):
    graph = create_graph()
    thread_id = thread_id or str(uuid4())
    
    initial_state = AgentState(
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
    
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": settings.recursion_limit}
    
    print(f"\n{'='*60}")
    print(f"Query: {query}")
    print(f"Thread ID: {thread_id}")
    print(f"{'='*60}\n")
    
    try:
        if verbose:
            async for event in graph.astream_events(initial_state, config=config, version="v2"):
                if event["event"] == "on_chain_start" and event["name"] in ["supervisor", "research", "analyst", "writer"]:
                    print(f">> [{event['name']}] Starting...")
                elif event["event"] == "on_chain_end" and event["name"] in ["supervisor", "research", "analyst", "writer"]:
                    output = _extract_output(event["data"].get("output", {}))
                    next_agent = output.get("next_agent", "?")
                    completed = output.get("completed_agents", [])
                    conf = output.get("confidence_score", 0)
                    print(f"OK [{event['name']}] Done -> Next: {next_agent} | Completed: {completed} | Confidence: {conf:.2f}")
        
        result = await graph.ainvoke(initial_state, config=config)
        
        print(f"\n{'='*60}")
        print("FINAL REPORT")
        print(f"{'='*60}\n")
        print(result.get("final_report", "No report generated"))
        print(f"\n{'='*60}")
        print(f"Confidence: {result.get('confidence_score', 0):.2f}")
        print(f"Completed Agents: {result.get('completed_agents', [])}")
        if result.get("error"):
            print(f"Error: {result['error']}")
        print(f"{'='*60}\n")
        return result
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(description="Multi-Agent Research System")
    parser.add_argument("query", nargs="?", help="Research query")
    parser.add_argument("--thread-id", help="Thread ID for continuing conversation")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--serve", action="store_true", help="Start API server")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    args = parser.parse_args()
    
    if args.serve:
        import uvicorn
        uvicorn.run("api.server:app", host="0.0.0.0", port=args.port, reload=True)
    elif args.query:
        asyncio.run(run_research(args.query, args.thread_id, args.verbose))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
