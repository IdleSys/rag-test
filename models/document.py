from datetime import datetime
from typing import Optional

from sqlmodel import Field, SQLModel


class DocumentModel(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    file_name: str
    url: str
    uploaded_at: datetime = Field(default_factory=datetime.now)
