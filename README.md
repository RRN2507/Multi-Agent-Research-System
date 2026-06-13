# 🤖 Multi-Agent Research & Report Generation System

> **Autonomously researches any topic, verifies claims, and delivers structured Markdown reports — powered by LangGraph Supervisor-Worker architecture.**


---

## 📌 What It Does

Send a query like `"Impact of AI on healthcare"` — the system spins up four specialised agents that:

1. **Research** the topic via live web search + local vector store
2. **Analyse** every claim, flag low-confidence findings, and score them
3. **Write** a clean, structured Markdown report
4. **Stream** progress in real-time via SSE or return a final JSON payload

No OpenAI key required. Runs entirely on **free-tier Groq + local embeddings**.

---

## 🏗️ Architecture

```
User Query
    │
    ▼
┌─────────────┐
│  SUPERVISOR │  ◄── LangGraph Command + goto routing
└──────┬──────┘
       │ routes dynamically based on shared AgentState
       │
   ┌───┴────────────────────────┐
   │                            │
   ▼                            ▼
┌──────────┐              ┌──────────┐
│ RESEARCH │              │ ANALYST  │
│  AGENT   │              │  AGENT   │
│          │              │          │
│ • Tavily │              │ • Claim  │
│   search │              │   verify │
│ • Chroma │              │ • Conf.  │
│   recall │              │   score  │
└────┬─────┘              └────┬─────┘
     │                         │
     └──────────┬──────────────┘
                │
                ▼
        ┌──────────────┐
        │    WRITER    │
        │    AGENT     │
        │              │
        │ • Structured │
        │   Markdown   │
        │   report     │
        └──────┬───────┘
               │
               ▼
             END
```

**Flow:** `Query → Supervisor → Research → Supervisor → Analyst → Supervisor → Writer → END`

The Supervisor never executes tasks itself — it only routes, tracks completion, and handles fallbacks.

---

## ✨ Key Features

| Feature | Detail |
|---|---|
| 🧠 **Supervisor-Worker Pattern** | LangGraph `Command + goto` — deterministic routing with fallback logic |
| 🔍 **Hybrid Retrieval** | Tavily live web search + ChromaDB local vector store |
| ✅ **Claim Verification** | Analyst flags uncertain findings with a `confidence_score` per claim |
| 📝 **Structured Reports** | Writer produces Pydantic-validated Markdown output |
| ⚡ **Free Inference** | Groq `llama-3.1-8b-instant` — no OpenAI costs |
| 🏠 **Local Embeddings** | `all-MiniLM-L6-v2` via sentence-transformers — no API key needed |
| 📡 **Dual API Mode** | `/invoke` (sync JSON) + `/stream` (SSE real-time progress) |
| 💾 **Checkpointing** | `LangGraph MemorySaver` — resume interrupted runs by thread ID |
| 🔭 **Observability** | LangSmith tracing for full agent step visibility |

---

## 🚀 Quick Start

### 1 — Clone & install

```bash
git clone https://github.com/RRN2507/Multi-Agent-Research-System.git
cd Multi-Agent-Research-System
pip install -r requirements.txt
```

### 2 — Configure API keys

```bash
cp .env.example .env
```

```env
# .env
GROQ_API_KEY=gsk_xxxxxxxxxxxxx
TAVILY_API_KEY=tvly_xxxxxxxxxxxxx

# Optional — LangSmith tracing
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=multi-agent-research
```

> Only **two keys** needed. Embeddings run locally — no paid vector DB.

### 3 — Run

```bash
# CLI — single query
python main.py "What is RAG in AI?" --verbose

# API server
python main.py --serve
# → http://localhost:8000
```

---

## 📡 API Reference

### `POST /invoke` — blocking, returns full report

```bash
curl -X POST http://localhost:8000/invoke \
  -H "Content-Type: application/json" \
  -d '{"query": "Impact of AI on healthcare", "stream": false}'
```

**Response**
```json
{
  "report": "## AI in Healthcare\n\n...",
  "confidence_score": 0.87,
  "completed_agents": ["research", "analyst", "writer"],
  "thread_id": "abc-123"
}
```

### `POST /stream` — real-time SSE progress

```bash
curl -X POST http://localhost:8000/stream \
  -H "Content-Type: application/json" \
  -d '{"query": "Future of autonomous vehicles"}'
```

Streams agent transitions live:
```
data: {"agent": "research", "status": "running"}
data: {"agent": "analyst",  "status": "running"}
data: {"agent": "writer",   "status": "done", "report": "..."}
```

### Other endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Service health check |
| `/threads/{id}/state` | GET | Retrieve checkpoint state by thread ID |

---

## 🔧 Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **Orchestration** | LangGraph 1.2 | StateGraph, Command routing, checkpointing |
| **LLM** | Groq `llama-3.1-8b-instant` | Fast, free inference |
| **Web Search** | Tavily API | Real-time web research |
| **Vector Store** | ChromaDB (local) | Persistent document retrieval |
| **Embeddings** | `all-MiniLM-L6-v2` | Local, no API key required |
| **API** | FastAPI + SSE | REST + streaming endpoints |
| **Validation** | Pydantic v2 | Structured agent outputs |
| **Observability** | LangSmith | End-to-end agent tracing |

---

## 📁 Project Structure

```
multi-agent-research/
├── agents/
│   ├── supervisor.py      # Command + goto routing logic
│   ├── research.py        # Tavily search + ChromaDB retrieval
│   ├── analyst.py         # Claim verification + confidence scoring
│   └── writer.py          # Structured Markdown report generation
├── graph/
│   └── builder.py         # StateGraph construction + compilation
├── api/
│   └── server.py          # FastAPI routes (/invoke, /stream, /health)
├── tools/
│   ├── tavily_tool.py     # Web search wrapper
│   └── chroma_tool.py     # Vector store read/write
├── prompts/               # Prompt templates per agent
├── config.py              # Pydantic BaseSettings (env management)
├── models.py              # AgentState TypedDict + output schemas
├── main.py                # CLI entry point + server launcher
├── requirements.txt
└── .env.example
```

---

## 🧠 State Schema

All agents share a single typed state object — no implicit side effects:

```python
class AgentState(TypedDict):
    messages:         Annotated[list[AnyMessage], add_messages]
    query:            str
    research_output:  str | None        # populated by Research agent
    analysis_output:  str | None        # populated by Analyst agent
    final_report:     str | None        # populated by Writer agent
    next_agent:       Literal["research", "analyst", "writer", "END", "START"]
    completed_agents: list[str]         # tracks routing history
    confidence_score: float             # 0.0 – 1.0, set by Analyst
    error:            str | None        # captured for fallback routing
    iteration_count:  int               # guards against infinite loops
```

---

## 💡 Design Decisions

**Why LangGraph over plain LangChain?**
StateGraph gives explicit control over agent transitions — the Supervisor always knows which agents have run and can deterministically decide what's next, rather than relying on an LLM to route.

**Why Groq instead of OpenAI?**
`llama-3.1-8b-instant` on Groq delivers sub-second latency at zero cost on the free tier — practical for a portfolio project that recruiters will actually clone and run.

**Why local embeddings?**
Eliminating the OpenAI embeddings API call removes a billing dependency and makes the project immediately runnable without a credit card.

---

## 📄 License

MIT — free to use for learning or production.

---

<p align="center">
  Built by <a href="https://github.com/RRN2507">Rushikesh R. Navale</a> · 
  <a href="https://linkedin.com/in/rrn2507">LinkedIn</a>
</p>
