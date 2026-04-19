from __future__ import annotations

import json

import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from advisor.config import Settings
from advisor.model_service import ModelRegistry


def main() -> None:
    settings = Settings.load()
    data = pd.read_csv(settings.data_dir / "data.csv")
    if "SalePrice" in data.columns:
        data = data.rename(columns={"SalePrice": "price"})

    selected_features = ["GrLivArea", "BedroomAbvGr", "FullBath", "YearBuilt", "Neighborhood"]
    dataset = data[selected_features + ["price"]].dropna()

    X_train, X_test, y_train, y_test = train_test_split(
        dataset[selected_features],
        dataset["price"],
        test_size=0.2,
        random_state=42,
    )

    registry = ModelRegistry(settings.artifacts_dir)
    results = {}
    for model_name, model in registry.models.items():
        predictions = model.predict(X_test)
        results[model_name] = {
            "mae": round(float(mean_absolute_error(y_test, predictions)), 2),
            "rmse": round(float(mean_squared_error(y_test, predictions, squared=False)), 2),
            "r2": round(float(r2_score(y_test, predictions)), 4),
        }

    settings.evaluation_output_path.parent.mkdir(parents=True, exist_ok=True)
    settings.evaluation_output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
