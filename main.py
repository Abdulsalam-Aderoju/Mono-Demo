import os
import sys
import json
import logging
import asyncio
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Optional

import joblib
import pandas as pd
import numpy as np
import shap

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# ==========================================
# 1. SETUP AUDIT LOGGING SYSTEM (JSON FORMAT)
# ==========================================
class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log_obj = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "message": record.getMessage(),
            "logger": record.name
        }
        if hasattr(record, "audit_data"):
            log_obj["audit"] = getattr(record, "audit_data")
        return json.dumps(log_obj)

# Create a dedicated auditor logger writing strictly to stdout
audit_logger = logging.getLogger("credit_decision_auditor")
audit_logger.setLevel(logging.INFO)
audit_logger.propagate = False  # Prevent propagating to root handlers

# Clear existing handlers to avoid duplicates
if audit_logger.handlers:
    audit_logger.handlers.clear()

stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(JSONFormatter())
audit_logger.addHandler(stream_handler)

# Initialize standard logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("app")

# ==========================================
# 2. FASTAPI SERVER INITIALIZATION
# ==========================================
app = FastAPI(
    title="Mono-Demo Credit Decisioning Engine",
    description="Production-ready credit decisioning engine with Calibrated LightGBM, SHAP explanation and fallback resiliency.",
    version="1.0.0"
)

# Enable CORS for local testing/integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# 3. ARTIFACT LOADING & SHAP EXPLAINER SETUP
# ==========================================
# Resiliently scan for artifacts folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
possible_dirs = [
    os.path.join(BASE_DIR, "Artifacts"),
    os.path.join(BASE_DIR, "artifacts"),
    "Artifacts",
    "artifacts"
]

artifacts_path = None
for p in possible_dirs:
    if os.path.exists(p) and os.path.isdir(p):
        artifacts_path = p
        break

if not artifacts_path:
    logger.error("Could not find Artifacts/ or artifacts/ directory in workspace.")
    sys.exit("Model artifacts directory missing.")

logger.info(f"Using artifacts directory: {artifacts_path}")

try:
    risk_model = joblib.load(os.path.join(artifacts_path, "risk_model.joblib"))
    tfidf_vectorizer = joblib.load(os.path.join(artifacts_path, "tfidf_vectorizer.joblib"))
    categorizer = joblib.load(os.path.join(artifacts_path, "categorizer.joblib"))
    logger.info("Successfully loaded all serialized model components.")
except Exception as e:
    logger.error(f"Error loading model artifacts: {e}")
    sys.exit("Critical error loading model artifacts.")

# Initialize SHAP TreeExplainer on base estimator
try:
    # risk_model is CalibratedClassifierCV. Extract base model of the first calibrated fold
    calibrated_classifier = risk_model.calibrated_classifiers_[0]
    # base_estimator attribute might be named base_estimator or estimator
    base_est = getattr(calibrated_classifier, "estimator", getattr(calibrated_classifier, "base_estimator", None))
    if base_est is None:
        raise AttributeError("Could not extract base estimator from calibrated classifier.")
    
    # Initialize the TreeExplainer
    explainer = shap.TreeExplainer(base_est)
    logger.info("Successfully initialized SHAP TreeExplainer on base LightGBM estimator.")
except Exception as e:
    logger.error(f"Error initializing SHAP TreeExplainer: {e}")
    explainer = None

# Initialize ThreadPoolExecutor for CPU-bound model predictions
executor = ThreadPoolExecutor(max_workers=8)

# ==========================================
# 4. REQUEST & RESPONSE SCHEMAS
# ==========================================
class ScoreRequest(BaseModel):
    income: float = Field(..., description="Monthly income of applicant in USD", ge=0)
    nsf_count: int = Field(..., description="Number of Non-Sufficient Funds events in past 90 days", ge=0)
    dsr: float = Field(..., description="Debt Service Ratio (total monthly debt payments / total monthly income)", ge=0)
    balance_volatility: float = Field(..., description="Standard deviation of daily balances", ge=0)
    loan_inflow_ratio: float = Field(..., description="Ratio of total loan inflows to total monthly income", ge=0)
    gambling_spend_index: float = Field(..., description="Proportion of total spend allocated to gambling merchants", ge=0, le=1)
    narration_risk_score: float = Field(..., description="Risk score extracted from transaction narration text lines", ge=0, le=1)

class SHAPContribution(BaseModel):
    feature: str
    shap_value: float

class ScoreResponse(BaseModel):
    probability: float = Field(..., description="Calculated probability of default")
    risk_tier: str = Field(..., description="Categorized risk tier (Low, Medium, High)")
    risk_band: str = Field(..., description="Risk tier band label, mapping to risk_tier")
    max_loan_affordable: int = Field(..., description="Maximum affordable loan limit in USD")
    top_features: List[SHAPContribution] = Field(..., description="Top 3 SHAP feature attributions")
    fallback: bool = Field(..., description="Flag indicating if the system used a hardcoded fallback decision")
    reason_code: Optional[str] = Field(None, description="Detailed system code explaining decision route")

# ==========================================
# 5. CORE INFERENCE WORKER
# ==========================================
def perform_inference(payload_dict: Dict[str, Any]) -> Dict[str, Any]:
    # Ensure correct feature ordering matches model
    features_list = list(risk_model.feature_names_in_)
    input_data = {feat: [payload_dict[feat]] for feat in features_list}
    df = pd.DataFrame(input_data)
    
    # 1. Predict uncalibrated default probability
    # CalibratedClassifierCV predict_proba returns probability for class 0, class 1
    probs = risk_model.predict_proba(df)
    prob_default = float(probs[0][1])
    
    # 2. Risk classification
    if prob_default < 0.2:
        risk_tier = "Low"
    elif prob_default < 0.5:
        risk_tier = "Medium"
    else:
        risk_tier = "High"
        
    # 3. Calculate max loan limit
    if prob_default > 0.7:
        max_loan = 0
    else:
        # Limit = Income * 0.3 * (1 - default_prob), rounded to the nearest 100
        raw_limit = payload_dict["income"] * 0.3 * (1.0 - prob_default)
        max_loan = int(round(raw_limit, -2))
        
    # 4. Extract SHAP values
    shap_contributions = []
    if explainer is not None:
        shap_vals = explainer.shap_values(df)
        # If shape is 2D (samples, features)
        if len(shap_vals.shape) == 2:
            row_shap = shap_vals[0]
        # Handle 3D shap values if returned by older versions (samples, classes, features)
        elif len(shap_vals.shape) == 3:
            row_shap = shap_vals[0][1]
        else:
            row_shap = shap_vals
            
        for name, val in zip(features_list, row_shap):
            shap_contributions.append({
                "feature": name,
                "shap_value": float(val)
            })
        # Sort by absolute magnitude descending
        shap_contributions.sort(key=lambda x: abs(x["shap_value"]), reverse=True)
    else:
        # Fallback empty SHAP values if explainer initialization failed
        shap_contributions = [{"feature": feat, "shap_value": 0.0} for feat in features_list]
        
    return {
        "probability": prob_default,
        "risk_tier": risk_tier,
        "risk_band": risk_tier,
        "max_loan_affordable": max(0, max_loan),
        "top_features": shap_contributions[:3],
        "fallback": False,
        "reason_code": "MODEL_DECISION_SUCCESS"
    }

# ==========================================
# 6. API ENDPOINTS
# ==========================================
@app.on_event("startup")
async def startup_event():
    # Warm up model and SHAP TreeExplainer
    try:
        dummy_payload = {
            "income": 5000.0,
            "nsf_count": 0,
            "dsr": 0.3,
            "balance_volatility": 500.0,
            "loan_inflow_ratio": 0.2,
            "gambling_spend_index": 0.0,
            "narration_risk_score": 0.1
        }
        logger.info("Warming up model and SHAP TreeExplainer...")
        # Run perform_inference directly to trigger Numba compilation and cache loading
        perform_inference(dummy_payload)
        logger.info("Warmup complete. System ready to serve traffic.")
    except Exception as e:
        logger.warning(f"Error during model warmup: {e}")

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model_version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "features_expected": list(risk_model.feature_names_in_)
    }

@app.post("/score", response_model=ScoreResponse)
async def score(payload: ScoreRequest):
    timestamp_start = datetime.utcnow().isoformat() + "Z"
    request_dict = payload.dict()
    
    try:
        # Enforce resilience with 2.0-second timeout
        # Using loop.run_in_executor to execute blocking Scikit-Learn/SHAP calls off the main loop
        result = await asyncio.wait_for(
            asyncio.get_running_loop().run_in_executor(executor, perform_inference, request_dict),
            timeout=2.0
        )
    except Exception as e:
        logger.warning(f"Inference exception triggered fallback path: {e}")
        # Hardcoded Resilient Fallback rule
        result = {
            "probability": 0.5,
            "risk_tier": "Medium-Manual-Review",
            "risk_band": "Medium-Manual-Review",
            "max_loan_affordable": 10000,
            "top_features": [
                {"feature": "SYSTEM_TIMEOUT_OR_FAILURE", "shap_value": 0.0}
            ],
            "fallback": True,
            "reason_code": "ERR_SYSTEM_FALLBACK"
        }
        
    # Structured JSON audit logging to stdout (for 90-day retraining)
    audit_data = {
        "input_payload": request_dict,
        "output_decision": result,
        "model_version": "1.0.0",
        "timestamp": timestamp_start
    }
    audit_logger.info("Credit Scoring Audit", extra={"audit_data": audit_data})
    
    return result

# ==========================================
# 7. STATIC FILES AND PAGE SERVING
# ==========================================
# Serve index.html directly on root endpoint
@app.get("/", response_class=HTMLResponse)
async def get_index():
    index_path = os.path.join(BASE_DIR, "static", "index.html")
    if not os.path.exists(index_path):
        return HTMLResponse(
            content="<html><body><h1>Mono-Demo Underwriter Dashboard</h1><p>Static index.html is missing. Please build the frontend.</p></body></html>",
            status_code=404
        )
    with open(index_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

# Mount general static assets directory if needed
static_dir = os.path.join(BASE_DIR, "static")
if not os.path.exists(static_dir):
    os.makedirs(static_dir)

app.mount("/static", StaticFiles(directory=static_dir), name="static")
