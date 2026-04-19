from __future__ import annotations

import json
from pathlib import Path
from typing import List

from advisor.schemas import RetrievedContext


def load_market_documents(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def format_context_snippets(items: List[RetrievedContext]) -> str:
    if not items:
        return (
            "No grounded market context was retrieved. Do not guess beyond the prediction model."
        )

    lines = []
    for item in items:
        lines.append(
            (
                f"{item.title}: {item.summary} "
                f"(market price/sqft={item.market_price_per_sqft:.0f}, "
                f"appreciation={item.yearly_appreciation_pct:.1f}%, "
                f"rental yield={item.rental_yield_pct:.1f}%, "
                f"risk={item.risk_level}, source={item.source})"
            )
        )
    return "\n".join(lines)
