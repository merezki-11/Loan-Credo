# import necessary libraries
import warnings
warnings.filterwarnings('ignore')
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib


# Load model, median and the encoder
model = joblib.load("model.pkl")
label_encoders = joblib.load("label_encoder.pkl")
medians = joblib.load("medians.pkl")
feature_names = joblib.load("feature_names.pkl")

app = FastAPI(title="Loan Credo", description="A machine learning powered service that predicts loan default risks.", version="1.0")
# Define input fields (raw — as if coming directly from CSV)
# This class is a data model used to represent information someone might provide when applying for a loan.
"""
if you notice when you look at the class you would see that next to the various information expected of the applicant
to fill that there is a data type, | and None = None. What this simply mean is that this information
when the customer fills it can either be the data type (float, str) or none i.e blank
"""
class LoanInput(BaseModel):
    # Categorical features (these were label encoded during training)
    loan_limit: str | None = None
    Gender: str | None = None
    approv_in_adv: str | None = None
    loan_type: str | None = None
    loan_purpose: str | None = None
    business_or_commercial: str | None = None
    interest_only: str | None = None
    occupancy_type: str | None = None
    credit_type: str | None = None
    co_applicant_credit_type: str | None = None
    submission_of_application: str | None = None

    #Numerical features
    loan_amount: float | None = None
    rate_of_interest: float | None = None
    Upfront_charges: float | None = None
    property_value: float | None = None
    income: float | None = None
    Credit_Score: float | None = None
    age: float | None = None
    LTV: float | None = None
    dtir1: float | None = None


@app.get("/")
def home():
    return {"message": "Welcome to the Loan Credo API!"}



@app.post("/predict")
def predict(data: LoanInput):
    try:
         # Convert input to DataFrame
         df = pd.DataFrame([data.dict()])

         # === Encode categorical features ===
         for col, encoder in label_encoders.items():
             if col in df.columns:
                 # Handle unseen categories gracefully
                 df[col] = df[col].fillna("Missing")
                 # Transform known values, use -1 for unknown
                 df[col] = df[col].apply(
                     lambda x: encoder.transform([x])[0] if x in encoder.classes_ else -1
                 )

         # === Missing Value Flagging ===
         df["missing_property_value"] = df["property_value"].isna().astype(int)
         df["missing_dtir"] = df["dtir1"].isna().astype(int)

         # === Fill missing values
         for col in df.columns:
            if not col.startswith("missing_"):
                df[col] = df[col].fillna(medians.get(col, 0))

         # === Ensure correct feature order ===
         df = df[feature_names]  # This reorders columns to match training

         # === Make prediction ===

         prediction = model.predict(df)[0]
         probability = model.predict_proba(df)[0].tolist()


         return {
             "prediction": int(prediction),
             "probability": probability,
             "risk_level": "High" if prediction == 1 else "Low"
         }
    except Exception as e:
         raise HTTPException(status_code=500, detail=str(e))


# Add a health check endpoint
@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None}


