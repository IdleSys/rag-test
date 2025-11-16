import chromadb
from chromadb.errors import NotFoundError
from sqlmodel import Session, create_engine

# ---------- SQL ----------
engine = create_engine("sqlite:///./db.sqlite3")

def get_db():
    with Session(engine) as db:
        yield db


# ---------- Chroma ----------

def get_chroma_client(persist_dir: str = "./chroma_db"):
    """
    Returns a Chroma PersistentClient (disk-based storage) using the new API.
    """
    client = chromadb.PersistentClient(
        path=persist_dir,  # directory where embeddings and metadata are stored
        # optional: turn off telemetry
        settings=chromadb.config.Settings(anonymized_telemetry=False)
    )
    return client


def get_or_create_chroma_collection(client, collection_name="documents"):
    """
    Returns a collection object. If it doesn't exist, creates it.
    """
    try:
        return client.get_collection(name=collection_name)
    except NotFoundError:
        return client.create_collection(name=collection_name)
