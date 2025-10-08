"""
REST API template for a web-based CHAP model.
Model developers should implement the actual train/predict logic.
"""

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, Optional
import io
import csv
import random

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel
import logging
logger = logging.getLogger(__name__)


app = FastAPI(
    title="Web-Based CHAP Model API",
    description="REST API for asynchronous model training and prediction",
    version="1.0.0",
)


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class JobInfo(BaseModel):
    job_id: str
    status: JobStatus
    created_at: datetime
    updated_at: datetime
    error_message: Optional[str] = None


class TrainResponse(BaseModel):
    job_id: str
    message: str


class PredictResponse(BaseModel):
    job_id: str
    message: str


# In-memory storage for jobs (in production, use database)
jobs: Dict[str, JobInfo] = {}
job_results: Dict[str, bytes] = {}  # Store prediction results


@app.post("/train", response_model=TrainResponse, summary="Start model training")
async def train(
    training_data: UploadFile = File(..., description="CSV file with training data"),
    model_name: str = Form(..., description="Name for the trained model"),
    polygons: Optional[UploadFile] = File(None, description="Optional GeoJSON file with polygons"),
    config: Optional[UploadFile] = File(None, description="Optional YAML configuration file"),
):
    """
    Start an asynchronous model training job.
    
    Parameters matching ExternalModel.train():
    - training_data: CSV file containing the training dataset
    - model_name: Identifier for the trained model
    - polygons: Optional GeoJSON file with geographic polygons
    - config: Optional YAML configuration file with model parameters
    
    Returns:
    - job_id: Unique identifier to track the training job
    """
    # Generate unique job ID
    logger.info(f"Received training request for model '{model_name}'")
    job_id = str(uuid.uuid4())
    
    # Create job entry
    job = JobInfo(
        job_id=job_id,
        status=JobStatus.PENDING,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )
    jobs[job_id] = job
    
    # TODO: Model developer should implement actual training logic here
    # This would typically:
    # 1. Save uploaded files to a working directory
    # 2. Start an async training process
    # 3. Update job status to RUNNING
    # 4. Update job status to COMPLETED/FAILED when done
    
    # For now, just read the files to ensure they're valid
    logger.info(f"Reading training data from {training_data.filename}")
    await training_data.read()
    if polygons:
        await polygons.read()
    if config:
        await config.read()
    
    # For testing: immediately mark job as completed
    job.status = JobStatus.COMPLETED
    job.updated_at = datetime.now(timezone.utc)
    
    logger.info(f"Training job '{job_id}' for model '{model_name}' completed successfully")
    return TrainResponse(
        job_id=job_id,
        message=f"Training job started for model '{model_name}'",
    )


@app.post("/predict", response_model=PredictResponse, summary="Start prediction")
async def predict(
    model_name: str = Form(..., description="Name of the trained model to use"),
    historic_data: UploadFile = File(..., description="CSV file with historic data"),
    future_data: UploadFile = File(..., description="CSV file with future data for prediction"),
    polygons: Optional[UploadFile] = File(None, description="Optional GeoJSON file with polygons"),
):
    """
    Start an asynchronous prediction job.
    
    Parameters matching ExternalModel.predict():
    - model_name: Identifier of the previously trained model
    - historic_data: CSV file containing historical data
    - future_data: CSV file containing future timepoints for prediction
    - polygons: Optional GeoJSON file with geographic polygons
    
    Returns:
    - job_id: Unique identifier to track the prediction job
    """
    # Generate unique job ID
    job_id = str(uuid.uuid4())
    
    # Create job entry
    job = JobInfo(
        job_id=job_id,
        status=JobStatus.PENDING,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )
    jobs[job_id] = job
    
    # TODO: Model developer should implement actual prediction logic here
    # This would typically:
    # 1. Save uploaded files to a working directory
    # 2. Load the trained model
    # 3. Start an async prediction process
    # 4. Update job status to RUNNING
    # 5. Save predictions to job_results when done
    # 6. Update job status to COMPLETED/FAILED
    
    # Read the future data to generate predictions based on it
    await historic_data.read()
    future_data_content = await future_data.read()
    if polygons:
        await polygons.read()
    
    # Parse the future data CSV to get locations and time periods
    try:
        # Decode the CSV content
        csv_text = future_data_content.decode('utf-8')
        csv_reader = csv.DictReader(io.StringIO(csv_text))
        
        # Generate mock predictions based on future data
        output_rows = []
        for row in csv_reader:
            # Extract location and time_period from the future data
            location = row.get('location', 'unknown')
            time_period = row.get('time_period', 'unknown')
            
            # Generate mock prediction samples
            # Base value between 50 and 200 for realistic disease case counts
            base_value = random.randint(50, 200)
            # Add some variation for each sample (±10%)
            sample_0 = int(base_value * random.uniform(0.9, 1.1))
            sample_1 = int(base_value * random.uniform(0.9, 1.1))
            sample_2 = int(base_value * random.uniform(0.9, 1.1))
            
            output_rows.append({
                'time_period': time_period,
                'location': location,
                'sample_0': sample_0,
                'sample_1': sample_1,
                'sample_2': sample_2
            })
        
        # Create CSV output
        output = io.StringIO()
        if output_rows:
            fieldnames = ['time_period', 'location', 'sample_0', 'sample_1', 'sample_2']
            writer = csv.DictWriter(output, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(output_rows)
            predictions_csv = output.getvalue().encode('utf-8')
        else:
            # If no rows in future data, return empty predictions with headers
            predictions_csv = b"time_period,location,sample_0,sample_1,sample_2\n"
        
        # Store the predictions for this job
        job_results[job_id] = predictions_csv
        
        logger.info(f"Generated {len(output_rows)} prediction rows for job {job_id}")
        
    except Exception as e:
        logger.error(f"Error processing future data: {e}")
        # Fall back to simple mock data if parsing fails
        mock_csv = b"time_period,location,sample_0,sample_1,sample_2\n2023-07,loc1,100,105,95\n2023-08,loc1,110,115,105\n2023-09,loc1,105,110,100"
        job_results[job_id] = mock_csv
    
    # Mark job as completed
    job.status = JobStatus.COMPLETED
    job.updated_at = datetime.now(timezone.utc)
    
    return PredictResponse(
        job_id=job_id,
        message=f"Prediction job started using model '{model_name}'",
    )


@app.get("/check_status/{job_id}", response_model=JobInfo, summary="Check job status")
async def check_status(job_id: str):
    """
    Check the status of a training or prediction job.
    
    Parameters:
    - job_id: The unique identifier returned by train or predict endpoints
    
    Returns:
    - Job information including current status
    """
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    
    return jobs[job_id]


@app.get("/fetch_predictions/{job_id}", summary="Fetch prediction results")
async def fetch_predictions(job_id: str):
    """
    Fetch the prediction results for a completed prediction job.
    
    Parameters:
    - job_id: The unique identifier returned by the predict endpoint
    
    Returns:
    - CSV file with predictions (if job is completed)
    """
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    
    job = jobs[job_id]
    
    if job.status != JobStatus.COMPLETED:
        raise HTTPException(
            status_code=400,
            detail=f"Job '{job_id}' is not completed. Current status: {job.status}",
        )
    
    if job_id not in job_results:
        raise HTTPException(
            status_code=404,
            detail=f"No results found for job '{job_id}'",
        )
    
    # Return the stored prediction results
    from fastapi.responses import Response
    return Response(
        content=job_results[job_id],
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=predictions_{job_id}.csv"},
    )


@app.get("/health", summary="Health check")
async def health_check():
    """Check if the API is running"""
    return {
        "status": "healthy",
        "total_jobs": len(jobs),
        "pending_jobs": sum(1 for j in jobs.values() if j.status == JobStatus.PENDING),
        "running_jobs": sum(1 for j in jobs.values() if j.status == JobStatus.RUNNING),
        "completed_jobs": sum(1 for j in jobs.values() if j.status == JobStatus.COMPLETED),
        "failed_jobs": sum(1 for j in jobs.values() if j.status == JobStatus.FAILED),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8005)