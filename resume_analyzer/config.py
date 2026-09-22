"""Settings, read from environment variables."""
import os
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Settings:
    gemini_api_key: Optional[str]
    gemini_model: str
    embed_model: str
    model_cache_dir: Optional[str]  # where the embedding model files live
    max_upload_mb: int
    max_jobs: int                   # job descriptions per /match-jobs request
    max_resumes: int                # resumes per /rank-resumes request


def load_settings() -> Settings:
    env = os.environ.get
    key = (env("GEMINI_API_KEY") or env("GOOGLE_API_KEY") or "").strip()
    return Settings(
        gemini_api_key=key or None,
        gemini_model=env("GEMINI_MODEL", "gemini-3.6-flash"),
        embed_model=env("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
        model_cache_dir=env("MODEL_CACHE_DIR") or None,
        max_upload_mb=int(env("MAX_UPLOAD_MB", "5")),
        max_jobs=int(env("MAX_JOBS", "20")),
        max_resumes=int(env("MAX_RESUMES", "25")),
    )
