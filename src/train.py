import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os

# Load dataset
data = pd.read_csv('../data/data.csv')

# Rename SalePrice to price if needed
if 'SalePrice' in data.columns:
    data.rename(columns={'SalePrice': 'price'}, inplace=True)

# Select only a few key features for simplicity
selected_features = ['GrLivArea', 'BedroomAbvGr', 'FullBath', 'YearBuilt', 'Neighborhood']
data = data[selected_features + ['price']].dropna()

# Split into X and y
X = data[selected_features]
y = data['price']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Identify numeric and categorical columns
numeric_cols = ['GrLivArea', 'BedroomAbvGr', 'FullBath', 'YearBuilt']
categorical_cols = ['Neighborhood']

# Preprocessing
numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_cols),
    ('cat', categorical_transformer, categorical_cols)
])

# Pipelines for models
lr_model = Pipeline([
    ('preprocessor', preprocessor),
    ('model', LinearRegression())
])

rf_model = Pipeline([
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Train models
lr_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)

# Evaluate
def evaluate(model):
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = mean_squared_error(y_test, preds, squared=False)
    r2 = r2_score(y_test, preds)
    return mae, rmse, r2

lr_mae, lr_rmse, lr_r2 = evaluate(lr_model)
rf_mae, rf_rmse, rf_r2 = evaluate(rf_model)

print(f"📊 Linear Regression: MAE={lr_mae:.2f}, RMSE={lr_rmse:.2f}, R²={lr_r2:.2f}")
print(f"🌲 Random Forest: MAE={rf_mae:.2f}, RMSE={rf_rmse:.2f}, R²={rf_r2:.2f}")

# Save models
os.makedirs('../artifacts', exist_ok=True)
joblib.dump(lr_model, '../artifacts/linear_regression.joblib')
joblib.dump(rf_model, '../artifacts/random_forest.joblib')

print("✅ Models saved successfully (simplified version).")