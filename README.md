**Multi-Agent Research & Report Generation System**



A production-grade Supervisor-Worker Multi-Agent System built with LangGraph that autonomously researches, analyzes, and generates structured reports on any topic.
🎯 Features
Supervisor Agent - Routes tasks using LangGraph Command + goto pattern
Research Agent - Tavily web search + ChromaDB local vector store
Analyst Agent - Verifies claims, flags low-confidence items, scores confidence
Writer Agent - Produces professional Markdown reports
Free Tier - Runs on Groq (llama-3.1-8b) + local embeddings
API - FastAPI with streaming (/invoke, /stream)
Observability - LangSmith tracing support
🏗️ Architecture
Query → Supervisor → Research → Supervisor → Analyst → Supervisor → Writer → END
🚀 Quick Start
# Clone
git clone https://github.com/RRN2507/Multi-Agent-Research-System.git
cd Multi-Agent-Research-System

# Install dependencies
pip install -r requirements.txt

# Configure API keys
cp .env.example .env
# Edit .env with your GROQ_API_KEY and TAVILY_API_KEY

# Run CLI
python main.py "What is RAG in AI?" --verbose

# Or start API server
python main.py --serve
📡 API Endpoints
Endpoint	Description
POST /invoke	Non-blocking, returns final report
POST /stream	Server-Sent Events for real-time progress
GET /health	Health check
GET /threads/{id}/state	Get checkpoint state
Example Request
curl -X POST http://localhost:8000/invoke \
  -H "Content-Type: application/json" \
  -d '{"query": "Impact of AI on healthcare", "stream": false}'
🔧 Tech Stack
LangGraph 1.2 - StateGraph, Command routing, checkpointing
Groq - llama-3.1-8b-instant (free, fast inference)
Tavily - Web search API
ChromaDB - Local vector store
sentence-transformers - all-MiniLM-L6-v2 embeddings (local, no API key)
FastAPI - REST + Streaming API
Pydantic v2 - Structured outputs & validation
📁 Project Structure
├── agents/          # Supervisor, Research, Analyst, Writer
├── graph/           # StateGraph builder
├── api/             # FastAPI server
├── tools/           # Tavily, ChromaDB tools
├── prompts/         # Prompt templates
├── config.py        # Settings management (Pydantic BaseSettings)
├── models.py        # State schema + Pydantic output models
└── main.py          # CLI entry point + streaming runner
⚙️ Configuration
Create .env file:

GROQ_API_KEY=gsk_xxxxxxxxxxxxx
TAVILY_API_KEY=tvly_xxxxxxxxxxxxx
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=multi-agent-research
🧠 State Schema
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    query: str
    research_output: str | None
    analysis_output: str | None
    final_report: str | None
    next_agent: Literal["research","analyst","writer","END","START"]
    completed_agents: list[str]
    confidence_score: float
    error: str | None
    iteration_count: int
📝 License
MIT License - feel free to use for learning or production.
