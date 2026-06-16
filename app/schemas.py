from pydantic import BaseModel, Field, field_validator
from typing import Literal


class CustomerFeatures(BaseModel):
    age: int = Field(ge=18, le=120, description="Customer age (18-120)")
    monthly_charges: float = Field(ge=0, description="Monthly service charges")
    total_charges: float = Field(ge=0, description="Total charges accumulated")
    tenure_months: int = Field(ge=0, le=600, description="Months as a customer")
    monthly_usage_gb: float = Field(ge=0, description="Monthly data usage in GB")
    customer_satisfaction: int = Field(ge=1, le=10, description="Satisfaction score (1-10)")
    number_of_services: int = Field(ge=0, le=20, description="Number of subscribed services")
    gender: Literal["male", "female"] = Field(description="Customer gender")
    contract_type: Literal["month-to-month", "one year", "two year"] = Field(
        description="Type of contract"
    )
    payment_method: Literal["electronic check", "mailed check",
                            "bank transfer", "credit card"] = Field(
        description="Payment method"
    )
    internet_service: Literal["dsl", "fiber optic", "no"] = Field(
        description="Internet service type"
    )
    phone_service: Literal["yes", "no"] = Field(description="Phone service active")
    streaming_tv: Literal["yes", "no"] = Field(description="Streaming TV active")
    streaming_movies: Literal["yes", "no"] = Field(description="Streaming movies active")

    @field_validator("age")
    @classmethod
    def age_must_be_reasonable(cls, v):
        if v < 18 or v > 120:
            raise ValueError("age must be between 18 and 120")
        return v

    @field_validator("customer_satisfaction")
    @classmethod
    def satisfaction_must_be_in_range(cls, v):
        if v < 1 or v > 10:
            raise ValueError("customer_satisfaction must be between 1 and 10")
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "age": 45,
                "monthly_charges": 85.50,
                "total_charges": 1200.00,
                "tenure_months": 24,
                "monthly_usage_gb": 25.0,
                "customer_satisfaction": 7,
                "number_of_services": 3,
                "gender": "male",
                "contract_type": "month-to-month",
                "payment_method": "electronic check",
                "internet_service": "fiber optic",
                "phone_service": "yes",
                "streaming_tv": "yes",
                "streaming_movies": "no"
            }
        }
    }


class ChurnPredictionResponse(BaseModel):
    churn_prediction: str = Field(description="Predicted churn outcome: 'yes' or 'no'")
    probability: float = Field(ge=0, le=1, description="Probability of churn (0-1)")


class CLVPredictionResponse(BaseModel):
    predicted_lifetime_value: float = Field(
        description="Predicted customer lifetime value"
    )


class HealthResponse(BaseModel):
    status: str = Field(description="API health status")
    model_loaded: bool = Field(description="Whether models are loaded")
    classifier: str = Field(description="Loaded classifier name")
    regressor: str = Field(description="Loaded regressor name")


class ErrorResponse(BaseModel):
    detail: str = Field(description="Error description")
