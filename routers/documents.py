import traceback
from typing import List

from fastapi import APIRouter, Depends, File, Response, UploadFile
from sqlmodel import Session

from constants import MessageType
from models import get_db
from schemas.base import EmptyResponseDTO, MessageDTO
from services.document_pipeline import proceess_document_pipeline
from services.file_service import FileService

router = APIRouter(prefix="", tags=["documents"])


@router.post("/upload")
async def upload_files(
    response: Response,
    session: Session = Depends(get_db),
    files: List[UploadFile] = File(...),
) -> EmptyResponseDTO:
    try:
        for file in files:
            file_path = FileService.upload_file(file)
            proceess_document_pipeline.handle(session, file)

        response.status_code = 201
        return EmptyResponseDTO(
            success=True,
            message=MessageDTO(
                type=MessageType.SUCCESS, text="Files Uploaded Correctly"
            ),
        )
    except Exception:
        # logger must log here
        print(traceback.format_exc())
        response.status_code = 500
        return EmptyResponseDTO(
            success=False,
            message=MessageDTO(
                type=MessageType.ERROR,
                text="There was an error while uploading the files. please try again later",
            ),
        )

