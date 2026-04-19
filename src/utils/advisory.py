from __future__ import annotations

from typing import Any, Dict, List

from utils.config import Settings


def generate_advisory(
    input_data: Dict[str, Any],
    predicted_price: float,
    insights: List[str],
    settings: Settings,
) -> Dict[str, str]:
    llm_reason = _generate_reason_with_llm(input_data, predicted_price, insights, settings)
    recommendation = _build_recommendation(input_data, predicted_price)

    if llm_reason is not None:
        reason = llm_reason
    else:
        reason = _build_rule_based_reason(input_data, predicted_price, insights)

    return {
        "recommendation": recommendation,
        "reason": reason,
    }


def _build_recommendation(input_data: Dict[str, Any], predicted_price: float) -> str:
    score = 0
    budget = float(input_data.get("budget", 0) or 0)

    if budget > 0 and predicted_price <= budget:
        score += 2
    elif budget > 0 and predicted_price > budget:
        score -= 2

    if int(input_data.get("size_sqft", 0) or 0) >= 1800:
        score += 1
    if int(input_data.get("year_built", 0) or 0) >= 2000:
        score += 1
    if int(input_data.get("bathrooms", 0) or 0) >= 2:
        score += 1
    if int(input_data.get("bhk", 0) or 0) >= 3:
        score += 1
    if int(input_data.get("garage_spaces", 0) or 0) >= 1:
        score += 1
    if bool(input_data.get("central_air", False)):
        score += 1
    if bool(input_data.get("basement_finished", False)):
        score += 1
    if int(input_data.get("overall_quality", 0) or 0) >= 7:
        score += 1
    if int(input_data.get("overall_condition", 0) or 0) >= 7:
        score += 1
    if str(input_data.get("functional_layout", "")).lower() == "poor":
        score -= 2
    if int(input_data.get("exterior_condition", 3) or 3) <= 2:
        score -= 1

    if score >= 6:
        return "Recommended"
    if score >= 2:
        return "Consider"
    return "Needs careful review"


def _build_rule_based_reason(
    input_data: Dict[str, Any],
    predicted_price: float,
    insights: List[str],
) -> str:
    notes: List[str] = [f"Predicted price is Rs {predicted_price:,.0f}."]
    budget = float(input_data.get("budget", 0) or 0)
    if budget > 0:
        if predicted_price <= budget:
            notes.append("The estimate is within the provided budget.")
        else:
            notes.append(f"The estimate is above budget by Rs {predicted_price - budget:,.0f}.")

    if insights:
        notes.append("Retrieved market insights highlight: " + " ".join(insights[:3]))
    else:
        notes.append("No relevant market insights were retrieved from the knowledge base.")

    return " ".join(notes)


def _generate_reason_with_llm(
    input_data: Dict[str, Any],
    predicted_price: float,
    insights: List[str],
    settings: Settings,
) -> str | None:
    if not settings.openai_api_key:
        return None

    try:
        from openai import OpenAI
    except ImportError:
        return None

    prompt = (
        "You are a grounded real-estate advisor. Use only the provided predicted price and "
        "retrieved insights. Do not invent facts. Write 3-4 sentences explaining the advice.\n\n"
        f"Input data: {input_data}\n"
        f"Predicted price: Rs {predicted_price:,.0f}\n"
        f"Retrieved insights: {insights}"
    )

    try:
        client = OpenAI(api_key=settings.openai_api_key)
        response = client.chat.completions.create(
            model=settings.openai_model,
            messages=[
                {
                    "role": "system",
                    "content": "Provide grounded explanations only from the supplied context.",
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=0.2,
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return None
