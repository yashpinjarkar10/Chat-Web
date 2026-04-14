from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    base_dir: Path
    groq_api_key: str | None
    mistral_api_key: str | None
    chroma_persist_dir: Path
    information_file: Path
    cors_allow_origins: list[str]


def _default_information_file(base_dir: Path) -> Path:
    # Prefer the "data/" folder if present, otherwise fall back to the existing root file.
    data_path = base_dir / "data" / "information.txt"
    if data_path.exists():
        return data_path
    return base_dir / "information.txt"


BASE_DIR = Path(__file__).resolve().parents[2]

settings = Settings(
    base_dir=BASE_DIR,
    groq_api_key=os.getenv("GROQ_API_KEY"),
    mistral_api_key=os.getenv("MistralAI"),
    chroma_persist_dir=Path(
        os.getenv("CHROMA_PERSIST_DIR", str(BASE_DIR / "db100" / "chroma_db100"))
    ),
    information_file=Path(os.getenv("INFORMATION_FILE", str(_default_information_file(BASE_DIR)))),
    cors_allow_origins=os.getenv("CORS_ALLOW_ORIGINS", "*").split(","),
)
