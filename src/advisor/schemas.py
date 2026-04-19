from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Literal, Optional, TypedDict


Intent = Literal["investment", "self-use"]


def _require_text(name: str, value: Any) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{name} must be a non-empty string")
    return text


def _require_int(name: str, value: Any, minimum: int, maximum: int) -> int:
    number = int(value)
    if number < minimum or number > maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return number


def _require_float(name: str, value: Any, minimum: float) -> float:
    number = float(value)
    if number <= minimum:
        raise ValueError(f"{name} must be greater than {minimum}")
    return number


@dataclass
class PropertyInput:
    city: str
    area: str
    property_type: str
    size_sqft: int
    bhk: int
    bathrooms: int
    year_built: int
    amenities: List[str] = field(default_factory=list)
    budget: float = 0.0
    user_intent: Intent = "investment"

    @classmethod
    def model_validate(cls, payload: Dict[str, Any]) -> "PropertyInput":
        amenities = payload.get("amenities", [])
        if isinstance(amenities, str):
            amenities = [item.strip() for item in amenities.split(",") if item.strip()]
        else:
            amenities = [str(item).strip() for item in amenities if str(item).strip()]

        user_intent = str(payload.get("user_intent", "")).strip()
        if user_intent not in {"investment", "self-use"}:
            raise ValueError("user_intent must be either 'investment' or 'self-use'")

        return cls(
            city=_require_text("city", payload.get("city")),
            area=_require_text("area", payload.get("area")),
            property_type=_require_text("property_type", payload.get("property_type")),
            size_sqft=_require_int("size_sqft", payload.get("size_sqft"), 200, 20000),
            bhk=_require_int("bhk", payload.get("bhk"), 1, 10),
            bathrooms=_require_int("bathrooms", payload.get("bathrooms"), 1, 10),
            year_built=_require_int("year_built", payload.get("year_built"), 1900, 2100),
            amenities=amenities,
            budget=_require_float("budget", payload.get("budget"), 0),
            user_intent=user_intent,
        )

    def model_dump(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RetrievedContext:
    doc_id: str
    title: str
    city: str
    area: str
    property_type: str
    summary: str
    market_price_per_sqft: float
    yearly_appreciation_pct: float
    rental_yield_pct: float
    risk_level: str
    source: str
    score: float

    @classmethod
    def model_validate(cls, payload: Dict[str, Any]) -> "RetrievedContext":
        return cls(**payload)

    def model_dump(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AdvisoryReport:
    predicted_price: float
    selected_model: str
    market_trend_summary: str
    investment_score: int
    risk_level: str
    recommendation: str
    justification: str
    valuation_gap_pct: float
    budget_fit: str
    comparable_market_price: Optional[float]
    user_intent: Intent
    confidence_note: str

    @classmethod
    def model_validate(cls, payload: Dict[str, Any]) -> "AdvisoryReport":
        return cls(**payload)

    def model_dump(self) -> Dict[str, Any]:
        return asdict(self)

    def as_readable_markdown(self) -> str:
        comparable = (
            f"Rs {self.comparable_market_price:,.0f}"
            if self.comparable_market_price is not None
            else "Not available"
        )
        return (
            "## Advisory Summary\n"
            f"- Predicted price: Rs {self.predicted_price:,.0f}\n"
            f"- Market comparable: {comparable}\n"
            f"- Investment score: {self.investment_score}/100\n"
            f"- Risk level: {self.risk_level}\n"
            f"- Recommendation: {self.recommendation}\n"
            f"- Budget fit: {self.budget_fit}\n"
            f"- Valuation gap: {self.valuation_gap_pct:.2f}%\n\n"
            "## Market Trend Summary\n"
            f"{self.market_trend_summary}\n\n"
            "## Justification\n"
            f"{self.justification}\n\n"
            "## Confidence Note\n"
            f"{self.confidence_note}"
        )


class AgentState(TypedDict, total=False):
    user_input: Dict[str, Any]
    normalized_input: Dict[str, Any]
    predicted_price: float
    selected_model: str
    model_predictions: Dict[str, float]
    retrieved_context: List[Dict[str, Any]]
    market_summary: str
    investment_score: int
    risk_level: str
    recommendation: str
    justification: str
    valuation_gap_pct: float
    comparable_market_price: Optional[float]
    budget_fit: str
    confidence_note: str
    final_report: Dict[str, Any]
