from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer
from torch import Tensor


def encode_to_embedding(document: Document | str) -> Tensor:
    content: str = ""

    match document:
        case str():
            content = document
        case Document():
            content = document.page_content

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embedding = model.encode(content)
    return embedding
