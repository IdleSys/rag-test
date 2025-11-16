import enum


class MessageType(enum.Enum):
    SUCCESS = "success"
    WARNING = "warning"
    ERROR = "error"
    INFO = "info"
