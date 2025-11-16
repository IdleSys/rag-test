import chromadb
from chromadb.api import ClientAPI
from chromadb.config import Settings
from sqlmodel import SQLModel

from models.document import (
    DocumentModel,
)

# -----SQL------
from .db import (  # noqa
    engine,
    get_chroma_client,
    get_db,
    get_or_create_chroma_collection,
)

SQLModel.metadata.create_all(engine)
