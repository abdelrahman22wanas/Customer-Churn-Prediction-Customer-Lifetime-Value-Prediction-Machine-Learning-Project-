from pydantic_settings import BaseSettings
from pathlib import Path
from typing import List


class Settings(BaseSettings):
    model_dir: Path = Path("models")
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_key: str = ""
    log_level: str = "INFO"
    cors_origins: List[str] = ["*"]

    model_config = {"env_file": ".env", "env_prefix": "CHURN_"}


settings = Settings()
