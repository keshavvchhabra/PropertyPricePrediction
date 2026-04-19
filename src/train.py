from __future__ import annotations

import os

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from advisor.config import Settings


def main() -> None:
    settings = Settings.load()
    data = pd.read_csv(settings.data_dir / "data.csv")

    if "SalePrice" in data.columns:
        data.rename(columns={"SalePrice": "price"}, inplace=True)

    selected_features = ["GrLivArea", "BedroomAbvGr", "FullBath", "YearBuilt", "Neighborhood"]
    data = data[selected_features + ["price"]].dropna()

    X = data[selected_features]
    y = data["price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    numeric_cols = ["GrLivArea", "BedroomAbvGr", "FullBath", "YearBuilt"]
    categorical_cols = ["Neighborhood"]

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
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
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
        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        rmse = mean_squared_error(y_test, preds, squared=False)
        r2 = r2_score(y_test, preds)
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
