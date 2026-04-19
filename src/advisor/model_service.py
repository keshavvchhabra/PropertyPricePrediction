from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import joblib
import pandas as pd

from advisor.schemas import PropertyInput


class ModelRegistry:
    def __init__(self, artifacts_dir: Path) -> None:
        self.models = {
            "linear_regression": joblib.load(artifacts_dir / "linear_regression.joblib"),
            "random_forest": joblib.load(artifacts_dir / "random_forest.joblib"),
        }

    def predict(self, property_input: PropertyInput, model_name: str) -> Tuple[float, Dict[str, float]]:
        features = pd.DataFrame(
            {
                "GrLivArea": [property_input.size_sqft],
                "BedroomAbvGr": [property_input.bhk],
                "FullBath": [property_input.bathrooms],
                "YearBuilt": [property_input.year_built],
                "Neighborhood": [property_input.area],
            }
        )

        predictions = {
            name: float(model.predict(features)[0]) for name, model in self.models.items()
        }
        selected_model = model_name if model_name in predictions else "random_forest"
        return predictions[selected_model], predictions
