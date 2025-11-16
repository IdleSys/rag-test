
from typing import Generic, List, Optional, TypeVar

from pydantic import NonNegativeInt
from pydantic.generics import GenericModel
from sqlmodel import SQLModel

from constants import MessageType

DataT = TypeVar("DataT")


class ResponseDTO(GenericModel, Generic[DataT]):
    success: bool
    message: Optional["MessageDTO"] = None
    data: Optional[DataT] = None


class EmptyResponseDTO(SQLModel):
    success: bool
    message: Optional["MessageDTO"] = None
    data: dict = {}


class PaginatedResponseDto(ResponseDTO, Generic[DataT]):
    page: int
    page_size: int
    total: int
    data: Optional[List[DataT]] = None


class PaginatedQueryParams(SQLModel):
    page: Optional[NonNegativeInt] = 1
    page_size: Optional[NonNegativeInt] = 10


class MessageDTO(SQLModel):
    type: MessageType
    text: str
