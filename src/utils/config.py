from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - bootstrap fallback
    def load_dotenv() -> bool:
        return False


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value else default


@dataclass(frozen=True)
class Settings:
    project_root: Path
    artifacts_dir: Path
    data_dir: Path
    knowledge_base_path: Path
    vector_db_path: Path
    evaluation_output_path: Path
    embedding_model: str
    retrieval_top_k: int
    default_model_name: str
    openai_api_key: str | None
    openai_model: str

    @classmethod
    def load(cls) -> "Settings":
        load_dotenv()
        project_root = Path(__file__).resolve().parents[2]
        artifacts_dir = Path(os.getenv("ARTIFACTS_DIR", project_root / "artifacts"))
        data_dir = Path(os.getenv("DATA_DIR", project_root / "data"))
        return cls(
            project_root=project_root,
            artifacts_dir=artifacts_dir,
            data_dir=data_dir,
            knowledge_base_path=Path(
                os.getenv("KNOWLEDGE_BASE_PATH", data_dir / "market_knowledge.json")
            ),
            vector_db_path=Path(
                os.getenv("VECTOR_DB_PATH", artifacts_dir / "vector_store")
            ),
            evaluation_output_path=Path(
                os.getenv("EVALUATION_OUTPUT_PATH", artifacts_dir / "evaluation.json")
            ),
            embedding_model=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
            retrieval_top_k=_env_int("RETRIEVAL_TOP_K", 3),
            default_model_name=os.getenv("DEFAULT_MODEL_NAME", "random_forest"),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        )
