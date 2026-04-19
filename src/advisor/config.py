from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional during bootstrap
    def load_dotenv() -> bool:
        return False


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value else default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return float(value) if value else default


@dataclass(frozen=True)
class Settings:
    project_root: Path
    artifacts_dir: Path
    data_dir: Path
    knowledge_base_path: Path
    evaluation_output_path: Path
    default_model_name: str
    retrieval_top_k: int
    undervalued_threshold: float
    overvalued_threshold: float

    @classmethod
    def load(cls) -> "Settings":
        load_dotenv()
        base_dir = Path(__file__).resolve().parents[2]
        artifacts_dir = Path(os.getenv("ARTIFACTS_DIR", base_dir / "artifacts"))
        data_dir = Path(os.getenv("DATA_DIR", base_dir / "data"))
        return cls(
            project_root=base_dir,
            artifacts_dir=artifacts_dir,
            data_dir=data_dir,
            knowledge_base_path=Path(
                os.getenv("KNOWLEDGE_BASE_PATH", data_dir / "market_knowledge.json")
            ),
            evaluation_output_path=Path(
                os.getenv("EVALUATION_OUTPUT_PATH", artifacts_dir / "evaluation.json")
            ),
            default_model_name=os.getenv("DEFAULT_MODEL_NAME", "random_forest"),
            retrieval_top_k=_env_int("RETRIEVAL_TOP_K", 3),
            undervalued_threshold=_env_float("UNDERVALUED_THRESHOLD", -0.08),
            overvalued_threshold=_env_float("OVERVALUED_THRESHOLD", 0.08),
        )
