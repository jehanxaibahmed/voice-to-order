"""Application settings, loaded from environment variables prefixed with ``VTO_``."""

from functools import lru_cache
from pathlib import Path
from typing import Annotated

from pydantic import AliasChoices, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="VTO_", env_file=".env", extra="ignore", populate_by_name=True
    )

    llm_model: str = "qwen2.5-coder:14b"
    llm_base_url: str | None = "http://localhost:5080/v1"
    llm_api_key: SecretStr | None = None
    llm_timeout_seconds: float = 120.0
    api_key_secret: str = "secret-key"

    deepgram_api_key: SecretStr | None = Field(
        default=None, validation_alias=AliasChoices("VTO_DEEPGRAM_API_KEY", "DEEPGRAM_API_KEY")
    )
    deepgram_model: str = "nova-3"
    openai_api_key: SecretStr | None = Field(
        default=None, validation_alias=AliasChoices("VTO_OPENAI_API_KEY", "OPENAI_API_KEY")
    )
    whisper_model: str = "whisper-1"
    assemblyai_api_key: SecretStr | None = Field(
        default=None, validation_alias=AliasChoices("VTO_ASSEMBLYAI_API_KEY", "ASSEMBLYAI_API_KEY")
    )
    assemblyai_base_url: str = "https://api.assemblyai.com"

    local_whisper_model: str = "small"
    local_whisper_compute_type: str = "int8"
    local_whisper_device: str = "auto"
    local_whisper_timeout_seconds: float = 300.0

    transcription_providers: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["deepgram", "whisper", "assemblyai"]
    )
    transcription_timeout_seconds: float = 60.0

    work_dir: Path = Path("./data")
    ffmpeg_binary: str = "ffmpeg"
    ffprobe_binary: str = "ffprobe"
    max_upload_bytes: int = 25 * 1024 * 1024

    @field_validator("transcription_providers", mode="before")
    @classmethod
    def _split_csv(cls, value: object) -> object:
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value


@lru_cache
def get_settings() -> Settings:
    return Settings()
