# --- Import necessary libraries ---
from fastapi import FastAPI, HTTPException  # FastAPI framework and error handling
from pydantic import BaseModel              # For data validation and request schema
import joblib                               # To load the pre-trained ML pipeline
import pandas as pd                         # To convert input data to DataFrame
from typing import List, Optional           # For typing support
from datetime import datetime               # For timestamping results
import uuid                                 # For generating unique result IDs

# --- Load the saved ML pipeline ---
pipeline = joblib.load("model/titanic_pipeline.pkl")  # Load trained pipeline
MODEL_VERSION = "1.0.0"                          # Set a version for the model
# Define the list of required input columns
REQUIRED_COLUMNS = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']

# --- Initialize the FastAPI app ---
app = FastAPI(title="Titanic Survival Predictor API")  # API title shown in docs

# --- Define the expected structure of input data ---
class Passenger(BaseModel):
    Pclass: int        # Passenger class (1st, 2nd, 3rd)
    Sex: str           # Gender (male/female)
    Age: float         # Age in years
    SibSp: int         # Number of siblings/spouses aboard
    Parch: int         # Number of parents/children aboard
    Fare: float        # Ticket fare
    Embarked: str      # Port of embarkation (C, Q, S)

# --- Define root endpoint to test the API is up ---
@app.get("/")
def read_root():
    return {"message": "Welcome to the Titanic Prediction API!"}  # Simple status check

# --- Define the prediction endpoint ---
@app.post("/predict")
def predict_survival(passengers: List[Passenger]):
    try:
        # Convert incoming list of Passenger objects to a pandas DataFrame
        input_df = pd.DataFrame([p.dict() for p in passengers])

        # Make predictions and get prediction probabilities
        preds = pipeline.predict(input_df)          # 0 = did not survive, 1 = survived
        probs = pipeline.predict_proba(input_df)    # Get probability of both classes

        # Package each prediction with additional metadata
        results = []
        for i in range(len(passengers)):
            results.append({
                "id": str(uuid.uuid4()),  # Unique ID for this prediction
                "prediction": int(preds[i]),  # Predicted class label
                "probability": {
                    "not_survived": round(probs[i][0], 3),  # Probability of class 0
                    "survived": round(probs[i][1], 3)        # Probability of class 1
                },
                "metadata": {
                    "timestamp": datetime.utcnow().isoformat(),  # Time of prediction
                    "model_version": MODEL_VERSION               # Version of the model used
                }
            })

        # Return all prediction results as JSON
        return {"results": results}

    except Exception as e:
        # Return HTTP 400 if something goes wrong (e.g., bad input)
        raise HTTPException(status_code=400, detail=str(e))

from fastapi import File, UploadFile  # For handling file uploads
import tempfile  # To create a temporary file to store uploaded data

# Endpoint to upload CSV file and get predictions
@app.post("/upload-csv")
async def upload_csv(file: UploadFile = File(...)):
    # Check if uploaded file has .csv extension
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")

    try:
        # Save uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
            tmp.write(await file.read())  # Write file content asynchronously
            tmp_path = tmp.name  # Save the path to read the file later

        # Read the CSV file into a DataFrame
        df = pd.read_csv(tmp_path)

        # Validate that required columns are present
        missing = set(REQUIRED_COLUMNS) - set(df.columns)
        if missing:
            raise HTTPException(status_code=400, detail=f"Missing columns: {missing}")

        # Run the trained ML pipeline on the input data
        raw_preds = pipeline.predict(df)  # Class predictions (0 or 1)
        proba = pipeline.predict_proba(df)  # Probabilities for each class

        # Build a list of results (each one contains prediction info + metadata)
        results = []
        for i in range(len(df)):
            result = {
                "id": str(uuid.uuid4()),  # Generate a unique ID for each prediction
                "prediction": int(raw_preds[i]),  # Predicted class (0 or 1)
                "probability": {
                    "not_survived": round(proba[i][0], 3),  # Probability for class 0
                    "survived": round(proba[i][1], 3)  # Probability for class 1
                },
                "metadata": {
                    "timestamp": datetime.utcnow().isoformat(),  # UTC timestamp
                    "model_version": MODEL_VERSION  # Current model version
                }
            }
            results.append(result)

        # Create a new DataFrame with predictions + input data
        result_df = df.copy()
        result_df["prediction"] = raw_preds
        result_df["prob_not_survived"] = proba[:, 0]
        result_df["prob_survived"] = proba[:, 1]
        result_df["model_version"] = MODEL_VERSION
        result_df["timestamp"] = datetime.utcnow()

        # Save the prediction results to a CSV file
        output_path = "prediction_output.csv"
        result_df.to_csv(output_path, index=False)

        # Return response with preview of predictions and download link
        return {
            "message": "Predictions completed!",
            "results": results[:5],  # Return only first 5 as preview
            "download": f"/download/{output_path}"  # Link to download full CSV
        }

    except Exception as e:
        # Return 500 error if anything fails
        raise HTTPException(status_code=500, detail=str(e))

from fastapi.responses import FileResponse  # To return file as response
import os  # Needed to check if file exists

# Endpoint to download the generated prediction CSV
@app.get("/download/{filename}")
def download_csv(filename: str):
    file_path = filename  # Get file path from URL
    if os.path.exists(file_path):  # Check if file exists
        return FileResponse(
            path=file_path,
            filename=filename,
            media_type='text/csv'  # Tell browser it's a CSV file
        )
    else:
        raise HTTPException(status_code=404, detail="File not found")
