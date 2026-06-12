from typing import Annotated, Any, Literal, TypedDict
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field, ConfigDict


class SupervisorDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    next_agent: Literal["research", "analyst", "writer", "END"] = Field(description="Next agent to route to, or END to terminate")
    reasoning: str = Field(description="Brief reasoning for the routing decision")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in this routing decision (0-1)")
    required_agents: list[Literal["research", "analyst", "writer"]] = Field(default_factory=list, description="Agents that still need to run")


class ResearchOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    summary: str = Field(description="Concise research summary")
    key_findings: list[str] = Field(description="List of key findings")
    sources: list[dict[str, str]] = Field(description="Sources with title, url, and relevance")
    gaps: list[str] = Field(default_factory=list, description="Identified information gaps needing further research")
    confidence: float = Field(ge=0.0, le=1.0, description="Overall research confidence")


class AnalysisOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    verified_claims: list[str] = Field(description="Claims verified with evidence")
    flagged_claims: list[dict[str, Any]] = Field(description="Claims with low confidence: {claim, issue, confidence}")
    contradictions: list[str] = Field(description="Identified contradictions")
    reasoning: str = Field(description="Analyst's reasoning process")
    confidence_score: float = Field(ge=0.0, le=1.0, description="Overall analysis confidence")
    needs_more_research: bool = Field(description="Whether more research is needed")


class WriterOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    markdown_report: str = Field(description="Complete markdown report")
    sections: dict[str, str] = Field(description="Report sections by name")


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    query: str
    research_output: str | None
    analysis_output: str | None
    final_report: str | None
    next_agent: Literal["research", "analyst", "writer", "END", "START"]
    completed_agents: list[str]
    confidence_score: float
    error: str | None
    iteration_count: int
