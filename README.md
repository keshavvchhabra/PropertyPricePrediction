# Property Price Prediction + RAG Advisory System

This project extends the existing CSV-based property price prediction system into a combined ML + RAG pipeline:

`User Input -> ML Prediction -> Retrieve Insights -> Generate Advisory Output`

The trained `joblib` models in `artifacts/` are reused directly. The CSV dataset remains unchanged. Market reasoning comes from a JSON knowledge base stored in `data/market_knowledge.json`.

## Project Structure

```text
PropertyPricePrediction/
├── artifacts/
│   ├── linear_regression.joblib
│   ├── random_forest.joblib
│   └── vector_store/
├── data/
│   ├── data.csv
│   └── market_knowledge.json
├── src/
│   ├── app.py
│   ├── evaluate.py
│   ├── train.py
│   ├── model/
│   │   └── predictor.py
│   ├── rag/
│   │   ├── knowledge_base.py
│   │   ├── retriever.py
│   │   └── vector_store.py
│   └── utils/
│       ├── advisory.py
│       ├── config.py
│       └── pipeline.py
├── .env.example
├── requirements.txt
└── README.md
```

## Step 1. Market Knowledge Base

The required JSON knowledge base is stored in [data/market_knowledge.json](/Users/keshav./Desktop/todayproject/PropertyPricePrediction/data/market_knowledge.json:1).

The loader extracts documents exactly as requested:

```python
documents = [item["insight"] for item in data]
```

## Step 2. ML Prediction Layer

[src/model/predictor.py](/Users/keshav./Desktop/todayproject/PropertyPricePrediction/src/model/predictor.py:1) loads the existing `linear_regression.joblib` and `random_forest.joblib` files and maps user input into the original model feature schema:

- `GrLivArea`
- `BedroomAbvGr`
- `FullBath`
- `YearBuilt`
- `Neighborhood`

## Step 3. RAG Layer

[src/rag/vector_store.py](/Users/keshav./Desktop/todayproject/PropertyPricePrediction/src/rag/vector_store.py:1) implements:

- JSON knowledge base loading
- sentence-transformers embeddings as the primary embedding path
- FAISS as the primary local vector index
- local persistence under `VECTOR_DB_PATH`
- similarity search for retrieval

To keep the app usable in offline bootstrap environments, the code falls back to TF-IDF + local cosine similarity if `sentence-transformers` or `faiss` is not available yet. After installing the updated requirements, the primary path uses sentence-transformers and FAISS.

## Step 4. Retrieval Function

[src/rag/retriever.py](/Users/keshav./Desktop/todayproject/PropertyPricePrediction/src/rag/retriever.py:1) exposes the required function:

```python
def retrieve_insights(query):
    # return top K relevant insights
```

## Step 5. ML + RAG Integration

[src/utils/pipeline.py](/Users/keshav./Desktop/todayproject/PropertyPricePrediction/src/utils/pipeline.py:1) implements:

```python
def predict_and_advise(input_data):
    predicted_price = model.predict(input_data)

    query = f"{input_data} property value factors"
    insights = retrieve_insights(query)

    return {
        "predicted_price": predicted_price,
        "insights": insights,
        "recommendation": "...",
        "reason": "..."
    }
```

## Step 6. Optional LLM Explanation

[src/utils/advisory.py](/Users/keshav./Desktop/todayproject/PropertyPricePrediction/src/utils/advisory.py:1) checks for `OPENAI_API_KEY`.

- If available, it asks the LLM for a grounded explanation using only the predicted price and retrieved insights.
- If not available, it uses a deterministic rule-based explanation.

## Step 7. Environment Configuration

Copy the example file and adjust values if needed:

```bash
cp .env.example .env
```

Supported variables:

- `VECTOR_DB_PATH`
- `EMBEDDING_MODEL`
- `RETRIEVAL_TOP_K`
- `ARTIFACTS_DIR`
- `DATA_DIR`
- `KNOWLEDGE_BASE_PATH`
- `DEFAULT_MODEL_NAME`
- `EVALUATION_OUTPUT_PATH`
- `OPENAI_API_KEY`
- `OPENAI_MODEL`

## Step 8. Run Instructions

Create and activate the virtual environment, then install dependencies:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Start the Streamlit app:

```bash
streamlit run src/app.py
```

Re-train models if needed:

```bash
python3 src/train.py
```

Evaluate the saved models:

```bash
python3 src/evaluate.py
```

## Step 9. Sample Input

```json
{
  "city": "Generic City",
  "area": "CollgCr",
  "size_sqft": 1500,
  "bhk": 3,
  "bathrooms": 2,
  "year_built": 2010,
  "budget": 250000,
  "overall_quality": 7,
  "overall_condition": 7,
  "garage_spaces": 1,
  "central_air": true,
  "basement_finished": false,
  "lot_area": 7000,
  "exterior_condition": 3,
  "functional_layout": "good"
}
```

## Step 10. Output Format

```json
{
  "predicted_price": 235532.01,
  "insights": [
    "Neighborhood quality is one of the strongest factors affecting property value.",
    "Poor exterior condition can significantly reduce property value.",
    "Larger living area (square footage) directly increases property price."
  ],
  "recommendation": "Recommended",
  "reason": "Predicted price is Rs 235,532. The estimate is within the provided budget. Retrieved market insights highlight: Neighborhood quality is one of the strongest factors affecting property value. Poor exterior condition can significantly reduce property value. Larger living area (square footage) directly increases property price."
}
```

## Evaluation Snapshot

Current metrics from the saved artifacts:

```json
{
  "linear_regression": {
    "mae": 26823.74,
    "rmse": 41238.16,
    "r2": 0.7783
  },
  "random_forest": {
    "mae": 21538.65,
    "rmse": 31403.22,
    "r2": 0.8714
  }
}
```
