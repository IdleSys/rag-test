from typing import Any, List

from sqlmodel import SQLModel

from models import get_chroma_client, get_or_create_chroma_collection
from utils.embedding import encode_to_embedding


class DocumentsChroma(SQLModel):
    ids: str | List[str]
    db_id: int
    embeddings: List[List[Any]]
    documents: List[Any]

    def save_document(self, collection_name: str = "default"):
        db = get_chroma_client()
        collection = get_or_create_chroma_collection(db, collection_name)
        collection.add(
            ids=self.ids,
            embeddings=self.embeddings,
            metadatas=[{"db_id": self.db_id} for _ in range(len(self.ids))],
            documents=self.documents,
        )

    @staticmethod
    def delete_document(cls, document_id: str, collection_name: str = "default"):
        db = get_chroma_client()
        collection = get_or_create_chroma_collection(db, collection_name)
        collection.delete(where={"metadata_field": {"$eq": document_id}})

    @staticmethod
    async def retrieve_top_k(
        query: str, top_k: int = 5, collection_name: str = "default"
    ) -> List[str]:
        db = get_chroma_client()
        collection = get_or_create_chroma_collection(db, collection_name)
        query_emb = encode_to_embedding(query).cpu().numpy().tolist()

        results = collection.query(query_embeddings=[query_emb], n_results=top_k)

        if results is not None:
            return results["documents"][0]  # type:ignore
        return []
