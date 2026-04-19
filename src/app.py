from __future__ import annotations

import json

import streamlit as st

from utils.pipeline import PropertyAdvisoryPipeline


@st.cache_resource
def load_pipeline() -> PropertyAdvisoryPipeline:
    return PropertyAdvisoryPipeline()


def run() -> None:
    """Main entry point for the Streamlit application."""
    st.set_page_config(page_title="Property Price Prediction + RAG Advisor", page_icon="🏠")
    st.title("Property Price Prediction + RAG Advisor")
    st.caption("User Input -> ML Prediction -> Retrieve Insights -> Generate Advisory Output")

    pipeline = load_pipeline()

    with st.sidebar:
        st.header("Property Input")
        city = st.text_input("City", value="Generic City")
        area = st.text_input("Area / Neighborhood", value="CollgCr")
        size_sqft = st.number_input("Living Area (sqft)", min_value=200, max_value=20000, value=1500)
        bhk = st.number_input("Bedrooms", min_value=1, max_value=10, value=3)
        bathrooms = st.number_input("Bathrooms", min_value=1, max_value=10, value=2)
        year_built = st.number_input("Year Built", min_value=1900, max_value=2100, value=2010)
        budget = st.number_input("Budget", min_value=0.0, value=250000.0, step=5000.0)
        overall_quality = st.slider("Overall Quality (optional)", min_value=1, max_value=10, value=6)
        overall_condition = st.slider("Overall Condition (optional)", min_value=1, max_value=10, value=6)
        garage_spaces = st.number_input("Garage Spaces (optional)", min_value=0, max_value=6, value=1)
        central_air = st.checkbox("Central Air", value=True)
        basement_finished = st.checkbox("Finished Basement", value=False)
        lot_area = st.number_input("Lot Area (optional)", min_value=0, max_value=100000, value=7000)
        exterior_condition = st.slider("Exterior Condition (1=poor, 5=excellent)", min_value=1, max_value=5, value=3)
        functional_layout = st.selectbox("Functional Layout", ["good", "average", "poor"])
        submit = st.button("Predict and Advise", type="primary")

    st.markdown(
        """
        The advisory pipeline reuses the trained CSV-based ML model for price prediction,
        then retrieves the most relevant market insights from `data/market_knowledge.json`.
        """
    )

    if not submit:
        st.info("Fill in the property details and run the pipeline.")
        return

    input_data = {
        "city": city,
        "area": area,
        "size_sqft": int(size_sqft),
        "bhk": int(bhk),
        "bathrooms": int(bathrooms),
        "year_built": int(year_built),
        "budget": float(budget),
        "overall_quality": int(overall_quality),
        "overall_condition": int(overall_condition),
        "garage_spaces": int(garage_spaces),
        "central_air": bool(central_air),
        "basement_finished": bool(basement_finished),
        "lot_area": int(lot_area),
        "exterior_condition": int(exterior_condition),
        "functional_layout": functional_layout,
    }

    result = pipeline.predict_and_advise(input_data)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Predicted Price", f"Rs {result['predicted_price']:,.0f}")
        st.subheader("Recommendation")
        st.write(result["recommendation"])

    with col2:
        st.subheader("Reason")
        st.write(result["reason"])

    st.subheader("Retrieved Insights")
    for insight in result["insights"]:
        st.write(f"- {insight}")

    st.subheader("Structured Output")
    st.code(json.dumps(result, indent=2), language="json")


if __name__ == "__main__":
    run()
