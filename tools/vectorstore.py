from langchain_chroma import Chroma
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_core.tools import tool
from langchain_core.documents import Document
from config import get_settings
from pydantic import BaseModel, Field
from typing import Optional


settings = get_settings()
_vectorstore = None


def get_vectorstore():
    global _vectorstore
    if _vectorstore is None:
        embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
        _vectorstore = Chroma(
            persist_directory=settings.chroma_persist_dir,
            embedding_function=embeddings,
            collection_name=settings.collection_name,
        )
    return _vectorstore


def initialize_vectorstore():
    return get_vectorstore()


def add_documents_to_vectorstore(documents: list[Document]) -> list[str]:
    vs = get_vectorstore()
    return vs.add_documents(documents)


class VectorstoreSearchInput(BaseModel):
    query: str = Field(description="Search query for relevant documents")
    k: int = Field(default=2, description="Number of documents to retrieve")
    filter: Optional[dict] = Field(default=None, description="Metadata filter")


@tool(args_schema=VectorstoreSearchInput)
def vectorstore_retriever_tool(query: str, k: int = 2, filter: Optional[dict] = None) -> list[dict]:
    """
    Retrieve relevant documents from the local ChromaDB vectorstore.
    Returns list of documents with content and metadata.
    """
    vs = get_vectorstore()
    docs = vs.similarity_search(query, k=k, filter=filter)
    return [{"content": doc.page_content[:500], "metadata": doc.metadata} for doc in docs]
