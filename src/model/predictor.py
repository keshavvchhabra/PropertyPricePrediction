from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import joblib
import pandas as pd


FEATURE_COLUMNS = ["GrLivArea", "BedroomAbvGr", "FullBath", "YearBuilt", "Neighborhood"]
NUMERIC_COLUMNS = ["GrLivArea", "BedroomAbvGr", "FullBath", "YearBuilt"]
CATEGORICAL_COLUMNS = ["Neighborhood"]
DEFAULT_MODEL_NAME = "random_forest"
DEFAULT_NEIGHBORHOOD = "CollgCr"


def build_feature_frame(input_data: Dict[str, Any]) -> pd.DataFrame:
    """Map user input to the exact feature schema expected by the saved models."""
    neighborhood = (
        input_data.get("neighborhood")
        or input_data.get("area")
        or input_data.get("Neighborhood")
        or DEFAULT_NEIGHBORHOOD
    )

    return pd.DataFrame(
        {
            "GrLivArea": [int(input_data.get("size_sqft", input_data.get("GrLivArea", 1500)))],
            "BedroomAbvGr": [int(input_data.get("bhk", input_data.get("BedroomAbvGr", 3)))],
            "FullBath": [int(input_data.get("bathrooms", input_data.get("FullBath", 2)))],
            "YearBuilt": [int(input_data.get("year_built", input_data.get("YearBuilt", 2010)))],
            "Neighborhood": [str(neighborhood)],
        }
    )


class PropertyPricePredictor:
    def __init__(self, artifacts_dir: Path) -> None:
        self.artifacts_dir = artifacts_dir
        self.models = {
            "linear_regression": joblib.load(artifacts_dir / "linear_regression.joblib"),
            "random_forest": joblib.load(artifacts_dir / "random_forest.joblib"),
        }

    def predict(self, input_data: Dict[str, Any], model_name: str = DEFAULT_MODEL_NAME) -> Dict[str, Any]:
        feature_frame = build_feature_frame(input_data)
        predictions = {
            name: float(model.predict(feature_frame)[0]) for name, model in self.models.items()
        }
        selected_model = model_name if model_name in predictions else DEFAULT_MODEL_NAME
        return {
            "predicted_price": predictions[selected_model],
            "selected_model": selected_model,
            "all_predictions": predictions,
        }
