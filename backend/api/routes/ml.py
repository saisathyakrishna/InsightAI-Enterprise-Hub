from fastapi import APIRouter, Depends
from pydantic import BaseModel
from typing import List
from services import ml_service
from services.rag_service import calculate_residual_correction
from core.security import get_current_user
from models.user import User

router = APIRouter(prefix="/ml", tags=["ML Forecasting Engine"])

class ForecastRequest(BaseModel):
    historical_sales: List[float]
    future_periods: int = 3
    project_id: int = 1  # Added this so it seamlessly passes into your RAG service

@router.post("/forecast")
def generate_forecast(
    data: ForecastRequest, 
    current_user: User = Depends(get_current_user)
):
    # 1. Run the baseline XGBoost prediction
    result = ml_service.train_and_forecast_demand(
        historical_sales=data.historical_sales,
        future_periods=data.future_periods
    )
    
    base_forecast_array = result.get("forecast", [])
    
    # 2. Extract the first predicted value to feed to the Gemini Agents
    baseline_value = base_forecast_array[0] if base_forecast_array else 0.0
    
    # 3. Trigger the Multi-Agent Committee
    llm_output = calculate_residual_correction(
        baseline_forecast=baseline_value,
        historical_sales=data.historical_sales,
        project_id=data.project_id
    )
    
    # 4. Extract the Adaptive Residual Correction (\epsilon_{LLM})
    epsilon = llm_output.get("residual_adjustment", 0.0)
    reasoning = llm_output.get("reasoning", "No qualitative adjustment applied.")
    
    # 5. Apply the adjustment to the final XGBoost array
    final_forecast = [val + epsilon for val in base_forecast_array]

    # 6. Return the comprehensive payload for your dashboard and Review-I paper
    return {
        "baseline_xgboost": base_forecast_array,
        "adaptive_forecast": final_forecast,
        "epsilon_llm": epsilon,
        "agent_reasoning": reasoning,
        "feature_importance": result.get("feature_importance", {}),
        "status": "success"
    }