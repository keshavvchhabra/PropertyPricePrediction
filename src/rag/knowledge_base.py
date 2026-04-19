from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple


def load_market_knowledge(path: Path) -> Tuple[List[dict], List[str]]:
    """Load the JSON knowledge base and expose documents for embedding."""
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    documents = [item["insight"] for item in data]
    return data, documents
