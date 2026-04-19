from __future__ import annotations

import json
from typing import Any, Dict

from model.predictor import PropertyPricePredictor
from rag.retriever import InsightRetriever
from utils.advisory import generate_advisory
from utils.config import Settings


class PropertyAdvisoryPipeline:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings.load()
        self.predictor = PropertyPricePredictor(self.settings.artifacts_dir)
        self.retriever = InsightRetriever(self.settings)

    def predict_and_advise(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        prediction = self.predictor.predict(
            input_data,
            model_name=self.settings.default_model_name,
        )

        query = f"{json.dumps(input_data, sort_keys=True)} property value factors"
        insights = self.retriever.retrieve_insights(query, top_k=self.settings.retrieval_top_k)
        advisory = generate_advisory(
            input_data=input_data,
            predicted_price=prediction["predicted_price"],
            insights=insights,
            settings=self.settings,
        )

        return {
            "predicted_price": prediction["predicted_price"],
            "insights": insights,
            "recommendation": advisory["recommendation"],
            "reason": advisory["reason"],
        }


def predict_and_advise(input_data: Dict[str, Any]) -> Dict[str, Any]:
    pipeline = PropertyAdvisoryPipeline()
    return pipeline.predict_and_advise(input_data)
