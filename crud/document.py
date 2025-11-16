from sqlmodel import Session, select

from models import DocumentModel


class DocumentCRUD:
    @staticmethod
    def create(session: Session, file_name: str, url: str) -> DocumentModel:
        doc = DocumentModel(file_name=file_name, url=url)
        session.add(doc)
        session.flush()
        session.refresh(doc)
        return doc

    @staticmethod
    def get(session: Session, doc_id: int) -> DocumentModel | None:
        return session.get(DocumentModel, doc_id)

    @staticmethod
    def list(session: Session):
        return session.exec(select(DocumentModel)).all()
