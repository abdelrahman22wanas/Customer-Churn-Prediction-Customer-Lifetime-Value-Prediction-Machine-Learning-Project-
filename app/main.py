"""
FastAPI application for Customer Churn & CLV Prediction.
Run with: uvicorn app.main:app
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings
from app.auth import APIKeyMiddleware
from app.schemas import (
    CustomerFeatures, ChurnPredictionResponse, CLVPredictionResponse,
    HealthResponse, ErrorResponse
)
from app.model_loader import load_models, preprocess_input
import logging

logging.basicConfig(level=getattr(logging, settings.log_level.upper()))
logger = logging.getLogger(__name__)

artifacts = None
classifier_name = "unknown"
regressor_name = "unknown"


@asynccontextmanager
async def lifespan(app):
    global artifacts, classifier_name, regressor_name
    try:
        artifacts = load_models()
        classifier_name = type(artifacts["classifier"]).__name__
        regressor_name = type(artifacts["regressor"]).__name__
        logger.info("Models loaded: classifier=%s, regressor=%s", classifier_name, regressor_name)
    except FileNotFoundError as e:
        logger.warning("No trained models found — %s", e)
        artifacts = None
    yield


app = FastAPI(
    title="Customer Churn & CLV Prediction API",
    description="ML API for predicting customer churn and lifetime value",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(APIKeyMiddleware)


@app.get("/health", response_model=HealthResponse)
def health_check():
    return HealthResponse(
        status="ok" if artifacts is not None else "degraded",
        model_loaded=artifacts is not None,
        classifier=classifier_name,
        regressor=regressor_name,
    )


@app.post(
    "/predict/churn",
    response_model=ChurnPredictionResponse,
    responses={400: {"model": ErrorResponse}, 403: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
)
def predict_churn(features: CustomerFeatures):
    if artifacts is None:
        raise HTTPException(status_code=503, detail="Models not loaded. Run train.py first.")

    try:
        X = preprocess_input(features.model_dump(), artifacts)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    classifier = artifacts["classifier"]
    target_encoder = artifacts["target_encoder"]

    pred_int = classifier.predict(X)[0]
    probability = classifier.predict_proba(X)[0, 1]
    churn_label = target_encoder.inverse_transform([pred_int])[0]

    return ChurnPredictionResponse(
        churn_prediction=str(churn_label),
        probability=round(float(probability), 4),
    )


@app.post(
    "/predict/clv",
    response_model=CLVPredictionResponse,
    responses={400: {"model": ErrorResponse}, 403: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
)
def predict_clv(features: CustomerFeatures):
    if artifacts is None:
        raise HTTPException(status_code=503, detail="Models not loaded. Run train.py first.")

    try:
        X = preprocess_input(features.model_dump(), artifacts)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    regressor = artifacts["regressor"]
    prediction = regressor.predict(X)[0]

    return CLVPredictionResponse(
        predicted_lifetime_value=round(float(prediction), 2),
    )
