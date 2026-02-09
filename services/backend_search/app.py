"""
Backend Search HTTP Service

Exposes KG search as an HTTP API for remote access from api_gateway.

Usage:
    uvicorn backend_search.app:app --host 0.0.0.0 --port 3003
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List

from .search import execute_search, SearchResult

app = FastAPI(title="Backend Search Service")


class SearchRequest(BaseModel):
    query: str
    tool: str = "wiki_idea_search"
    top_k: int = 5
    domains: Optional[List[str]] = None


class SearchResponse(BaseModel):
    success: bool
    content: str
    results_count: int
    latency_ms: int
    error: Optional[str] = None


@app.get("/health")
async def health():
    return {"status": "healthy", "service": "backend_search"}


@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    result = await execute_search(
        query=request.query,
        tool=request.tool,
        top_k=request.top_k,
        domains=request.domains,
    )
    return SearchResponse(
        success=result.success,
        content=result.content,
        results_count=result.results_count,
        latency_ms=result.latency_ms,
        error=result.error,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3003)
