from fastapi import APIRouter

from schemas.query import QueryRequest
from services.rag_pipeline import run_rag_pipeline

router = APIRouter(prefix="", tags=["query"])


@router.post("/query")
async def query_route(payload: QueryRequest):
    return await run_rag_pipeline(payload.query, (payload.top_k or 5))

