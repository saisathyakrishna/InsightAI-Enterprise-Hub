import xgboost as xgb
import shap
import pandas as pd
import numpy as np
from typing import List, Dict, Any

def train_and_forecast_demand(historical_sales: List[float], future_periods: int = 3) -> Dict[str, Any]:
    # 1. Prepare lagged features from time-series data
    df = pd.DataFrame({"sales": historical_sales})
    df["lag_1"] = df["sales"].shift(1)
    df["lag_2"] = df["sales"].shift(2)
    df = df.dropna()
    
    if len(df) < 2:
        mean_val = float(np.mean(historical_sales)) if historical_sales else 0.0
        return {
            "forecast": [mean_val] * future_periods,
            "explanation": "Insufficient historical data for complex lags; returning moving average baseline."
        }

    X = df[["lag_1", "lag_2"]]
    y = df["sales"]
    
    # 2. Train XGBoost Regressor
    model = xgb.XGBRegressor(n_estimators=50, learning_rate=0.1)
    model.fit(X, y)
    
    # 3. Generate recursive future predictions
    predictions = []
    last_known = list(historical_sales[-2:])
    for _ in range(future_periods):
        next_features = np.array([[last_known[-1], last_known[-2]]])
        next_pred = float(model.predict(next_features)[0])
        predictions.append(next_pred)
        last_known.append(next_pred)
        
    # 4. Generate SHAP values for model transparency
    explainer = shap.Explainer(model)
    shap_values = explainer(X)
    
    # Extract mean absolute SHAP values to indicate feature importance
    feature_importance = dict(zip(["lag_1", "lag_2"], [float(np.abs(shap_values.values).mean(axis=0)[0]), float(np.abs(shap_values.values).mean(axis=0)[1])]))

    return {
        "forecast": predictions,
        "feature_importance": feature_importance,
        "status": "success"
    }