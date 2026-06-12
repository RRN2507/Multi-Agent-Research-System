
import os
from functools import lru_cache
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    groq_api_key: str = Field(..., validation_alias="GROQ_API_KEY")
    tavily_api_key: str = Field(..., validation_alias="TAVILY_API_KEY")
    langchain_api_key: str | None = Field(None, validation_alias="LANGCHAIN_API_KEY")
    langchain_tracing_v2: bool = Field(True, validation_alias="LANGCHAIN_TRACING_V2")
    langchain_project: str = Field("multi-agent-research", validation_alias="LANGCHAIN_PROJECT")
    
    # Groq models (free, fast)
    supervisor_model: str = "llama-3.1-8b-instant"
    worker_model: str = "llama-3.1-8b-instant"
    temperature: float = 0.1
    max_tokens: int = 4000
    
    chroma_persist_dir: str = "./chroma_db"
    embedding_model: str = "text-embedding-3-small"
    collection_name: str = "research_docs"
    tavily_max_results: int = 5
    tavily_search_depth: str = "advanced"
    recursion_limit: int = 25
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 1

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
