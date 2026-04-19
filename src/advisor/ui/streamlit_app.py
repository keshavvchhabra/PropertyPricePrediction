from __future__ import annotations

import json

import streamlit as st

from advisor.agent import RealEstateAdvisorAgent
from advisor.schemas import AdvisoryReport


def run() -> None:
    st.set_page_config(
        page_title="AI Real Estate Advisory Assistant",
        page_icon="🏠",
        layout="wide",
    )
    st.title("AI Real Estate Advisory Assistant")
    st.caption(
        "Predict property prices, ground recommendations in local market context, and generate a structured advisory report."
    )

    agent = RealEstateAdvisorAgent()

    with st.sidebar:
        st.header("Property Details")
        city = st.text_input("City", value="Ames")
        area = st.text_input("Area / Neighborhood", value="CollgCr")
        property_type = st.selectbox("Property Type", ["flat", "villa", "independent house", "townhouse"])
        size_sqft = st.number_input("Size (sqft)", min_value=200, max_value=20000, value=1500)
        bhk = st.number_input("BHK", min_value=1, max_value=10, value=3)
        bathrooms = st.number_input("Bathrooms", min_value=1, max_value=10, value=2)
        year_built = st.number_input("Year Built", min_value=1900, max_value=2100, value=2010)
        budget = st.number_input("Budget", min_value=10000.0, value=250000.0, step=5000.0)
        user_intent = st.radio("Intent", ["investment", "self-use"], horizontal=True)
        amenities = st.multiselect(
            "Amenities",
            ["parking", "gym", "security", "clubhouse", "park", "pool", "school access"],
            default=["parking", "security"],
        )
        submit = st.button("Generate Advisory Report", type="primary")

    st.markdown(
        """
        ### Guardrails
        - Retrieval is grounded only on the local knowledge base.
        - The advisory engine is instructed not to guess when market data is missing.
        - Final recommendations combine the trained ML prediction with retrieved market signals.
        """
    )

    if not submit:
        st.info("Fill in the property details and generate a report.")
        return

    raw_input = {
        "city": city,
        "area": area,
        "property_type": property_type,
        "size_sqft": int(size_sqft),
        "bhk": int(bhk),
        "bathrooms": int(bathrooms),
        "year_built": int(year_built),
        "amenities": amenities,
        "budget": float(budget),
        "user_intent": user_intent,
    }

    report_payload = agent.run(raw_input)
    report = AdvisoryReport.model_validate(report_payload)

    col1, col2, col3 = st.columns(3)
    col1.metric("Predicted Price", f"Rs {report.predicted_price:,.0f}")
    comparable = (
        f"Rs {report.comparable_market_price:,.0f}"
        if report.comparable_market_price is not None
        else "Unavailable"
    )
    col2.metric("Market Comparable", comparable)
    col3.metric("Investment Score", f"{report.investment_score}/100")

    left, right = st.columns([1.2, 1])
    with left:
        st.subheader("Readable Advisory")
        st.markdown(report.as_readable_markdown())

    with right:
        st.subheader("Structured Report")
        st.json(report_payload, expanded=True)

    st.subheader("Export JSON")
    st.code(json.dumps(report_payload, indent=2), language="json")
