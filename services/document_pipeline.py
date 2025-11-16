from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generator, List, Optional

from fastapi import UploadFile
from langchain_core.documents import Document
from pydantic import ValidationError
from sqlmodel import Session

from conf import settings
from crud.document import DocumentCRUD
from services.chroma_document_service import DocumentsChroma
from utils.embedding import encode_to_embedding
from utils.markdown_splitter import split_markdown


@dataclass
class DocumentContext:
    id: Optional[int] = None
    url: Optional[str] = None
    file_path: Path = Path()
    chunks: List[Document] = field(default_factory=list)
    embeddings: Optional[Generator[Any, None, None]] = None

class DocumentBaseHandler:
    def __init__(self, next_handler=None) -> None:
        self.next_handler = next_handler

    def handle(
        self,
        session: Session,
        file_obj: UploadFile,
        ctx: DocumentContext = DocumentContext(),
    ):
        if self.next_handler:
            return self.next_handler.handle(session, file_obj, ctx)
        return file_obj


class SaveToSQLHandler(DocumentBaseHandler):
    def handle(
        self,
        session: Session,
        file_obj: UploadFile,
        ctx: DocumentContext = DocumentContext(),
    ):
        file_name = file_obj.filename
        file_url = settings.BASE_DIR/"upload"/file_name
        if file_name is None:
            raise ValidationError("Uploaded file has no name")

        print("FUCK", ctx)
        new_doc = DocumentCRUD.create(
            session = session,
            file_name=file_name,
            url= str(file_url)
        )

        ctx.id = new_doc.id
        ctx.url = new_doc.url
        ctx.file_path =file_url 

        return super().handle(session, file_obj, ctx)


class ChunkHandler(DocumentBaseHandler):
    def handle(
        self,
        session: Session,
        file_obj: UploadFile,
        ctx: DocumentContext = DocumentContext(),
    ):
        content = ctx.file_path.read_text()
        chunks = split_markdown(content)

        ctx.chunks = chunks
        return super().handle(session, file_obj, ctx)


class EmbedHandler(DocumentBaseHandler):
    def embeddings(self, chunks: List[Document]):
        for chunk in chunks:
            yield encode_to_embedding(chunk).tolist()


    def handle(
        self,
        session: Session,
        file_obj: UploadFile,
        ctx: DocumentContext = DocumentContext(),
    ):
        ctx.embeddings = self.embeddings(ctx.chunks)
        return super().handle(session, file_obj, ctx)


class SaveToChromahandler(DocumentBaseHandler):
    def handle(
        self,
        session: Session,
        file_obj: UploadFile,
        ctx: DocumentContext = DocumentContext(),
    ):
        if ctx.embeddings is None or ctx.id is None:
            raise ValidationError(
                "Chroma handler needs embeddings and id to be passed in the context"
            )

        ids = [f"{file_obj.filename}_chunk_{i}" for i in range(len(ctx.chunks))]
        embeddings = list(ctx.embeddings)
        new_chroma_doc = DocumentsChroma(
            ids=ids,
            documents=[x.page_content for x in ctx.chunks],
            embeddings=embeddings, #type:ignore
            db_id=ctx.id,
        )
        new_chroma_doc.save_document()
        return super().handle(session, file_obj, ctx)


proceess_document_pipeline = SaveToSQLHandler(
    ChunkHandler(EmbedHandler(SaveToChromahandler()))
)
