import sys
from pathlib import Path


class Settings:
    BASE_DIR = Path(__file__).parent
    UPLOAD_DIR = BASE_DIR / "upload"

    def __init__(self) -> None:
        "Bootstrap project with the give settings"
        sys.path.append(str(self.BASE_DIR))
        self.UPLOAD_DIR.mkdir(exist_ok=True)


settings = Settings()
