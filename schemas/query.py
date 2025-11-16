from typing import List, Optional

from sqlmodel import Field, SQLModel


class QueryRequest(SQLModel):
    query: str = Field(..., description="User question")
    top_k: int|None = Field(5, description="Number of relevant chunks to retrieve")


class QueryResponse(SQLModel):
    answer: Optional[str] = Field(None, description="Final vetted answer")
    context_used: List[str] = Field(
        default_factory=list, description="Retrieved chunks"
    )
    refused: bool = Field(False, description="Whether the system refused to answer")
    message: Optional[str] = Field(None, description="Refusal or status message")
