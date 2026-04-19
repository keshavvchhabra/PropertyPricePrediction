from __future__ import annotations

from advisor.schemas import AdvisoryReport, PropertyInput


class ReportBuilder:
    def build(self, user_input: PropertyInput, payload: dict) -> AdvisoryReport:
        return AdvisoryReport(
            predicted_price=payload["predicted_price"],
            selected_model=payload["selected_model"],
            market_trend_summary=payload["market_summary"],
            investment_score=payload["investment_score"],
            risk_level=payload["risk_level"],
            recommendation=payload["recommendation"],
            justification=payload["justification"],
            valuation_gap_pct=payload["valuation_gap_pct"],
            budget_fit=payload["budget_fit"],
            comparable_market_price=payload["comparable_market_price"],
            user_intent=user_input.user_intent,
            confidence_note=payload["confidence_note"],
        )
