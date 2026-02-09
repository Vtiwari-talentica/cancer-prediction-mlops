"""FastAPI service for cancer prediction"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import joblib
from datetime import datetime
from typing import List, Dict, Any
import logging

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logger import setup_logger
from src.utils.config import Config
from src.utils.helpers import load_model
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from fastapi.responses import Response

# Setup
logger = setup_logger(__name__, "logs/api.log")
config = Config("configs/config.yaml")
app = FastAPI(
    title="Cancer Prediction API",
    description="ML-powered cancer survival prediction service",
    version="2.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prometheus Metrics
PREDICTION_COUNTER = Counter('predictions_total', 'Total predictions made', ['model_version', 'outcome'])
PREDICTION_LATENCY = Histogram('prediction_latency_seconds', 'Prediction latency')
MODEL_VERSION = Gauge('model_version_info', 'Model version information', ['version', 'timestamp'])
DRIFT_SCORE = Gauge('feature_drift_score', 'Feature drift score', ['feature'])

# Global state
model = None
preprocessor = None
feature_names = None
model_metadata = {}


class PatientData(BaseModel):
    """Request schema for prediction"""
    Age: int = Field(..., ge=0, le=120, description="Patient age")
    Gender: str = Field(..., description="Gender (Male/Female)")
    Country: str
    Cancer_Stage: str = Field(..., description="Cancer stage (Localized/Regional/Metastatic)")
    Family_History: str = Field(..., description="Family history (Yes/No)")
    Smoking_History: str
    Alcohol_Consumption: str
    Obesity_BMI: str
    Diet_Risk: str
    Physical_Activity: str
    Diabetes: str
    Inflammatory_Bowel_Disease: str
    Screening_History: str
    Genetic_Mutation: str
    Treatment_Type: str
    Tumor_Size_mm: float = Field(..., gt=0, description="Tumor size in mm")
    Early_Detection: str
    Healthcare_Access: str
    Insurance_Status: str
    Urban_or_Rural: str
    Economic_Classification: str
    Healthcare_Costs: float = Field(..., ge=0)
    Incidence_Rate_per_100K: float = Field(..., ge=0, le=100)
    Mortality_Rate_per_100K: float = Field(..., ge=0, le=100)
    Survival_5_years: str
    
    @validator('Gender', 'Family_History', 'Smoking_History', 'Alcohol_Consumption', 
               'Diabetes', 'Inflammatory_Bowel_Disease', 'Early_Detection', 'Survival_5_years')
    def validate_binary(cls, v):
        if v not in ['Yes', 'No', 'Male', 'Female']:
            raise ValueError(f'Must be Yes/No or Male/Female, got {v}')
        return v
    
    @validator('Cancer_Stage')
    def validate_stage(cls, v):
        if v not in ['Localized', 'Regional', 'Metastatic']:
            raise ValueError(f'Invalid cancer stage: {v}')
        return v


class PredictionResponse(BaseModel):
    """Response schema"""
    prediction: int = Field(..., description="0 = Not Survived, 1 = Survived")
    probability: float = Field(..., ge=0, le=1, description="Survival probability")
    confidence: str = Field(..., description="High/Medium/Low")
    risk_factors: List[str]
    model_version: str
    timestamp: str


class BatchPredictionRequest(BaseModel):
    patients: List[PatientData]


@app.on_event("startup")
async def startup_event():
    """Load model and preprocessor on startup"""
    global model, preprocessor, feature_names, model_metadata
    
    logger.info("Starting Cancer Prediction API...")
    
    try:
        # Load model
        model_path = Path(config.get('paths.models')) / "best_model.pkl"
        if not model_path.exists():
            # Fallback to any available model
            model_files = list(Path(config.get('paths.models')).glob("*_model.pkl"))
            if model_files:
                model_path = model_files[0]
                logger.warning(f"Using fallback model: {model_path}")
        
        model = load_model(str(model_path))
        logger.info(f"Model loaded from {model_path}")
        
        # Load preprocessor
        preprocessor_path = Path("models/artifacts/preprocessors.pkl")
        preprocessor = joblib.load(preprocessor_path)
        logger.info("Preprocessor loaded")
        
        # Get feature names from processed data
        sample_data = pd.read_csv("data/processed/train_processed.csv", nrows=1)
        feature_names = [col for col in sample_data.columns if col != 'Survival_Prediction']
        
        # Metadata
        model_metadata = {
            'version': '2.0.0',
            'model_type': model_path.stem.replace('_model', ''),
            'loaded_at': datetime.now().isoformat(),
            'num_features': len(feature_names)
        }
        
        MODEL_VERSION.labels(
            version=model_metadata['version'],
            timestamp=model_metadata['loaded_at']
        ).set(1)
        
        logger.info(f"API ready - Model: {model_metadata['model_type']}, Features: {len(feature_names)}")
        
    except Exception as e:
        logger.error(f"Startup failed: {str(e)}")
        raise


@app.get("/")
def root():
    """Root endpoint"""
    return {
        "service": "Cancer Prediction API",
        "version": model_metadata.get('version', 'unknown'),
        "status": "healthy",
        "endpoints": {
            "/predict": "Single prediction",
            "/predict/batch": "Batch predictions",
            "/health": "Health check",
            "/metrics": "Prometheus metrics",
            "/model/info": "Model information"
        }
    }


@app.get("/health")
def health_check():
    """Health check endpoint"""
    if model is None or preprocessor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "preprocessor_loaded": preprocessor is not None,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/metrics")
def metrics():
    """Prometheus metrics endpoint"""
    return Response(content=generate_latest(), media_type="text/plain")


@app.get("/model/info")
def model_info():
    """Get model information"""
    return {
        "metadata": model_metadata,
        "num_features": len(feature_names) if feature_names else 0,
        "feature_names": feature_names[:10] if feature_names else []  # First 10
    }


@app.post("/predict", response_model=PredictionResponse)
@PREDICTION_LATENCY.time()
async def predict(patient: PatientData):
    """Single prediction endpoint"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert to DataFrame
        data_dict = patient.dict()
        df = pd.DataFrame([data_dict])
        
        # Apply feature engineering
        from src.features.feature_engineering import FeatureEngineer
        engineer = FeatureEngineer()
        df = engineer.engineer_features(df)
        
        # Preprocess
        from src.data.preprocessor import DataPreprocessor
        prep = DataPreprocessor()
        prep.scalers = preprocessor.get('scalers', {})
        prep.encoders = preprocessor.get('encoders', {})
        
        df_processed = prep.preprocess_pipeline(df, fit=False, include_target=False)
        
        # Align features
        for col in feature_names:
            if col not in df_processed.columns:
                df_processed[col] = 0
        df_processed = df_processed[feature_names]
        
        # Predict
        prediction = int(model.predict(df_processed)[0])
        probability = float(model.predict_proba(df_processed)[0][1])
        
        # Confidence
        if probability > 0.8 or probability < 0.2:
            confidence = "High"
        elif probability > 0.6 or probability < 0.4:
            confidence = "Medium"
        else:
            confidence = "Low"
        
        # Risk factors
        risk_factors = []
        if data_dict['Cancer_Stage'] == 'Metastatic':
            risk_factors.append("Advanced cancer stage")
        if data_dict['Family_History'] == 'Yes':
            risk_factors.append("Family history present")
        if data_dict['Smoking_History'] == 'Yes':
            risk_factors.append("Smoking history")
        if data_dict['Tumor_Size_mm'] > 50:
            risk_factors.append("Large tumor size")
        if data_dict['Early_Detection'] == 'No':
            risk_factors.append("Late detection")
        
        # Update metrics
        PREDICTION_COUNTER.labels(
            model_version=model_metadata['version'],
            outcome='survived' if prediction == 1 else 'not_survived'
        ).inc()
        
        logger.info(f"Prediction: {prediction}, Probability: {probability:.3f}, Confidence: {confidence}")
        
        return PredictionResponse(
            prediction=prediction,
            probability=probability,
            confidence=confidence,
            risk_factors=risk_factors,
            model_version=model_metadata['version'],
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch")
async def predict_batch(request: BatchPredictionRequest):
    """Batch prediction endpoint"""
    predictions = []
    
    for patient in request.patients:
        try:
            result = await predict(patient)
            predictions.append(result.dict())
        except Exception as e:
            predictions.append({"error": str(e)})
    
    return {
        "predictions": predictions,
        "total": len(predictions),
        "timestamp": datetime.now().isoformat()
    }


@app.post("/admin/trigger-drift-detection")
async def trigger_drift_detection(background_tasks: BackgroundTasks):
    """Trigger drift detection job"""
    async def run_drift_detection():
        logger.info("Running drift detection...")
        # Import and run drift detection
        from src.monitoring.drift_detection import DriftDetector
        detector = DriftDetector()
        report = detector.detect_drift()
        logger.info(f"Drift detection complete: {report}")
    
    background_tasks.add_task(run_drift_detection)
    return {"status": "Drift detection triggered", "timestamp": datetime.now().isoformat()}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
