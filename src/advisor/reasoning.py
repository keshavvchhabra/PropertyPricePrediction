from __future__ import annotations

from statistics import mean
from typing import List, Tuple

from advisor.schemas import PropertyInput, RetrievedContext


class InvestmentReasoner:
    def __init__(self, undervalued_threshold: float, overvalued_threshold: float) -> None:
        self.undervalued_threshold = undervalued_threshold
        self.overvalued_threshold = overvalued_threshold

    def analyze(
        self,
        user_input: PropertyInput,
        predicted_price: float,
        contexts: List[RetrievedContext],
    ) -> dict:
        comparable_price = None
        valuation_gap_pct = 0.0
        trend_summary = self._build_market_summary(contexts)

        if contexts:
            comparable_price = mean(
                item.market_price_per_sqft * user_input.size_sqft for item in contexts
            )
            valuation_gap_pct = ((predicted_price - comparable_price) / comparable_price) * 100

        score, recommendation, risk_level = self._score(
            user_input=user_input,
            predicted_price=predicted_price,
            comparable_price=comparable_price,
            contexts=contexts,
            valuation_gap_pct=valuation_gap_pct,
        )

        budget_fit = (
            "within budget"
            if predicted_price <= user_input.budget
            else f"above budget by Rs {predicted_price - user_input.budget:,.0f}"
        )

        if not contexts:
            justification = (
                "Prediction completed with the existing ML model, but no matching grounded market "
                "documents were retrieved. Recommendation is conservative and avoids unsupported claims."
            )
            confidence_note = "Low confidence: market context unavailable, so trend reasoning is limited."
        else:
            justification = self._build_justification(
                user_input=user_input,
                predicted_price=predicted_price,
                comparable_price=comparable_price,
                valuation_gap_pct=valuation_gap_pct,
                contexts=contexts,
            )
            confidence_note = (
                "Grounded on retrieved market documents only. If local market conditions changed "
                "recently, refresh the knowledge base before relying on this advisory output."
            )

        return {
            "market_summary": trend_summary,
            "investment_score": score,
            "recommendation": recommendation,
            "risk_level": risk_level,
            "valuation_gap_pct": valuation_gap_pct,
            "comparable_market_price": comparable_price,
            "budget_fit": budget_fit,
            "justification": justification,
            "confidence_note": confidence_note,
        }

    def _score(
        self,
        user_input: PropertyInput,
        predicted_price: float,
        comparable_price: float | None,
        contexts: List[RetrievedContext],
        valuation_gap_pct: float,
    ) -> Tuple[int, str, str]:
        score = 50
        avg_appreciation = mean(item.yearly_appreciation_pct for item in contexts) if contexts else 0.0
        avg_yield = mean(item.rental_yield_pct for item in contexts) if contexts else 0.0

        if comparable_price is not None:
            if valuation_gap_pct <= self.undervalued_threshold * 100:
                score += 20
            elif valuation_gap_pct >= self.overvalued_threshold * 100:
                score -= 20

        if user_input.user_intent == "investment":
            if avg_appreciation >= 6:
                score += 12
            if avg_yield >= 3.5:
                score += 10
        else:
            if predicted_price <= user_input.budget:
                score += 10
            if avg_appreciation > 0:
                score += 4

        if predicted_price > user_input.budget:
            score -= 15

        risk_hits = sum(1 for item in contexts if item.risk_level.lower() == "high")
        score -= risk_hits * 8
        score = max(0, min(100, score))

        if score >= 75:
            recommendation = "Strong buy"
            risk_level = "Low"
        elif score >= 60:
            recommendation = "Consider buy"
            risk_level = "Medium"
        elif score >= 45:
            recommendation = "Hold / negotiate"
            risk_level = "Medium"
        else:
            recommendation = "Avoid for now"
            risk_level = "High"

        return score, recommendation, risk_level

    @staticmethod
    def _build_market_summary(contexts: List[RetrievedContext]) -> str:
        if not contexts:
            return "No relevant market trend documents were found in the local knowledge base."

        summary_parts = []
        for item in contexts:
            summary_parts.append(
                (
                    f"{item.area}, {item.city}: {item.summary} "
                    f"Avg price Rs {item.market_price_per_sqft:,.0f}/sqft, "
                    f"appreciation {item.yearly_appreciation_pct:.1f}%, "
                    f"rental yield {item.rental_yield_pct:.1f}%, "
                    f"risk {item.risk_level.lower()}."
                )
            )
        return " ".join(summary_parts)

    @staticmethod
    def _build_justification(
        user_input: PropertyInput,
        predicted_price: float,
        comparable_price: float | None,
        valuation_gap_pct: float,
        contexts: List[RetrievedContext],
    ) -> str:
        if comparable_price is None:
            return "No comparable market price could be derived from the retrieved context."

        avg_appreciation = mean(item.yearly_appreciation_pct for item in contexts)
        avg_yield = mean(item.rental_yield_pct for item in contexts)
        intent_clause = (
            "Investment focus prioritizes appreciation and rental yield."
            if user_input.user_intent == "investment"
            else "Self-use focus prioritizes affordability and medium-term market stability."
        )
        valuation_clause = (
            "The property appears undervalued versus retrieved market comparables."
            if valuation_gap_pct < 0
            else "The property appears priced above retrieved market comparables."
        )

        return (
            f"Predicted price is Rs {predicted_price:,.0f} against an estimated comparable value of "
            f"Rs {comparable_price:,.0f}, creating a valuation gap of {valuation_gap_pct:.2f}%. "
            f"Retrieved locations show average appreciation of {avg_appreciation:.1f}% and rental yield "
            f"of {avg_yield:.1f}%. {valuation_clause} {intent_clause}"
        )
