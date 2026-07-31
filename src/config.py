from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT_DIR / ".env", extra="ignore")

    embeddings_device_id: int
    data_raw_dir: Path = ROOT_DIR / "data" / "raw"
    data_converted_dir: Path = ROOT_DIR / "data" / "converted"


settings = Settings()
