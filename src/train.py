from __future__ import annotations

import os

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from model.predictor import CATEGORICAL_COLUMNS, FEATURE_COLUMNS, NUMERIC_COLUMNS
from utils.config import Settings


def main() -> None:
    settings = Settings.load()
    data = pd.read_csv(settings.data_dir / "data.csv")

    if "SalePrice" in data.columns:
        data = data.rename(columns={"SalePrice": "price"})

    dataset = data[FEATURE_COLUMNS + ["price"]].dropna()
    X = dataset[FEATURE_COLUMNS]
    y = dataset["price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    numeric_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        [
            ("num", numeric_transformer, NUMERIC_COLUMNS),
            ("cat", categorical_transformer, CATEGORICAL_COLUMNS),
        ]
    )

    lr_model = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("model", LinearRegression()),
        ]
    )
    rf_model = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("model", RandomForestRegressor(n_estimators=100, random_state=42)),
        ]
    )

    lr_model.fit(X_train, y_train)
    rf_model.fit(X_train, y_train)

    def evaluate(model: Pipeline) -> tuple[float, float, float]:
        predictions = model.predict(X_test)
        mae = mean_absolute_error(y_test, predictions)
        rmse = root_mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        return mae, rmse, r2

    lr_mae, lr_rmse, lr_r2 = evaluate(lr_model)
    rf_mae, rf_rmse, rf_r2 = evaluate(rf_model)
    print(f"Linear Regression: MAE={lr_mae:.2f}, RMSE={lr_rmse:.2f}, R2={lr_r2:.2f}")
    print(f"Random Forest: MAE={rf_mae:.2f}, RMSE={rf_rmse:.2f}, R2={rf_r2:.2f}")

    os.makedirs(settings.artifacts_dir, exist_ok=True)
    joblib.dump(lr_model, settings.artifacts_dir / "linear_regression.joblib")
    joblib.dump(rf_model, settings.artifacts_dir / "random_forest.joblib")


if __name__ == "__main__":
    main()
