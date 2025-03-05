"""
main.py

FastAPI application for the Monument Recognition system.
Handles concurrent image processing requests from Flutter frontend.
"""

import os
import json
import uuid
import asyncio
import numpy as np
from typing import Optional, List, Dict, Any
from concurrent.futures import ThreadPoolExecutor

import cv2
from fastapi import FastAPI, File, Form, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import the MonumentPipeline directly
from src.pipeline import MonumentPipeline

# Create FastAPI app
app = FastAPI(
    title="Monument Recognition API",
    description="API for detecting and recognizing monuments in images",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Initialize global variables for the pipeline
DETECTOR_MODEL_PATH = os.environ.get("DETECTOR_MODEL_PATH", "./weights/yolov11_det.engine")
DETECTOR_CONFIDENCE = float(os.environ.get("DETECTOR_CONFIDENCE", "0.7"))
EXTRACTOR_MODEL_NAME = os.environ.get("EXTRACTOR_MODEL_NAME", "efficientnet_b3")
MILVUS_URI = os.environ.get("MILVUS_URI", "./data/monumentdb.db")
COLLECTION_NAME = os.environ.get("COLLECTION_NAME", "global_features")
DEVICE = os.environ.get("DEVICE", None)  # None will auto-select cuda if available

# Max number of concurrent processing tasks
MAX_CONCURRENT_TASKS = int(os.environ.get("MAX_CONCURRENT_TASKS", "4"))

# Response models
class Match(BaseModel):
    landmark: str
    filename: str
    score: float

class Detection(BaseModel):
    bbox: List[int]
    confidence: float
    matches: List[Match]

class RecognitionResponse(BaseModel):
    request_id: str
    num_detections: int
    detections: List[Detection]
    location: Optional[str] = None

# Initialize thread pool for concurrent processing
executor = ThreadPoolExecutor(max_workers=MAX_CONCURRENT_TASKS)

# Track active tasks
active_tasks: Dict[str, asyncio.Task] = {}

# Initialize pipeline
monument_pipeline = None

try:
    monument_pipeline = MonumentPipeline(
        detector_model_path=DETECTOR_MODEL_PATH,
        detector_confidence=DETECTOR_CONFIDENCE,
        extractor_model_name=EXTRACTOR_MODEL_NAME,
        milvus_uri=MILVUS_URI,
        collection_name=COLLECTION_NAME,
        device=DEVICE
    )
    print("Monument recognition pipeline initialized successfully")
except Exception as e:
    print(f"Failed to initialize monument pipeline: {str(e)}")
    raise

@app.get("/")
async def root():
    return {"message": "Monument Recognition API is running"}

@app.get("/health")
async def health_check():
    if monument_pipeline is None:
        raise HTTPException(status_code=503, detail="Monument recognition pipeline not initialized")
    
    # Show number of active tasks
    return {
        "status": "healthy", 
        "pipeline": "initialized",
        "active_tasks": len(active_tasks),
        "max_concurrent_tasks": MAX_CONCURRENT_TASKS
    }

def process_image_task(img_array, location):
    """
    Process image in a separate thread to avoid blocking the main event loop.
    This function runs in a ThreadPoolExecutor.
    """
    try:
        # Process the image with the pipeline
        results = monument_pipeline.process_image(
            image=img_array,
            top_k=3,
            return_image=False
        )
        
        # Add location to response if provided
        if location:
            results['location'] = location
            
        return results
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        raise

@app.post("/recognize", response_model=RecognitionResponse)
async def recognize_monument(
    background_tasks: BackgroundTasks,
    image: UploadFile = File(...),
    location: Optional[str] = Form(None),
):
    """
    Recognize monuments in an uploaded image.
    
    Parameters:
    - image: Image file to analyze
    - location: Optional location information to aid recognition
    
    Returns:
    - JSON with detection and recognition results and unique request ID
    """
    global monument_pipeline, active_tasks
    
    if monument_pipeline is None:
        raise HTTPException(status_code=503, detail="Monument recognition pipeline not initialized")
    
    # Check if we're at capacity for concurrent tasks
    if len(active_tasks) >= MAX_CONCURRENT_TASKS:
        raise HTTPException(
            status_code=429, 
            detail="Too many concurrent requests. Please try again later."
        )
    
    try:
        # Generate a unique ID for this request
        request_id = str(uuid.uuid4())
        
        # Read image file
        contents = await image.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image format")
        
        # Process the image in a thread pool
        task = asyncio.create_task(
            asyncio.to_thread(process_image_task, img, location)
        )
        
        # Add to active tasks
        active_tasks[request_id] = task
        
        # Wait for the task to complete
        results = await task
        
        # Add request ID to results
        results['request_id'] = request_id
        
        # Remove from active tasks
        if request_id in active_tasks:
            del active_tasks[request_id]
        
        return results
        
    except Exception as e:
        # Clean up the task if it exists
        if 'request_id' in locals() and request_id in active_tasks:
            del active_tasks[request_id]
        
        print(f"Error processing image: {str(e)}")
        raise HTTPException(status_code=503, detail=f"Error processing image: {str(e)}")

@app.post("/recognize/visualize")
async def recognize_and_visualize(
    background_tasks: BackgroundTasks,
    image: UploadFile = File(...),
    location: Optional[str] = Form(None),
):
    """
    Recognize monuments and return with visualization.
    
    Parameters:
    - image: Image file to analyze
    - location: Optional location information to aid recognition
    
    Returns:
    - JSON with results and base64 encoded image with visualizations
    """
    # Reuse the main recognition endpoint
    results = await recognize_monument(background_tasks, image, location)
    
    try:
        # Need to read the image again since the file pointer was consumed
        await image.seek(0)
        contents = await image.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image format")
        
        # Visualize the results
        vis_img = monument_pipeline.visualize_results(
            image=img.copy(),
            results=results,
            show_all_matches=False
        )
        
        # Encode the visualized image to base64
        import base64
        _, buffer = cv2.imencode('.jpg', vis_img)
        img_str = base64.b64encode(buffer).decode('utf-8')
        
        # Add visualization to response
        results_dict = results.dict() if hasattr(results, 'dict') else results
        results_dict['visualization'] = f"data:image/jpeg;base64,{img_str}"
        
        return results_dict
        
    except Exception as e:
        print(f"Error creating visualization: {str(e)}")
        raise HTTPException(status_code=503, detail=f"Error creating visualization: {str(e)}")

@app.post("/recognize/json")
async def recognize_monument_json(
    background_tasks: BackgroundTasks,
    image: UploadFile = File(...),
    location: Optional[str] = Form(None),
):
    """
    Recognize monuments and return raw JSON results.
    
    Parameters:
    - image: Image file to analyze
    - location: Optional location information to aid recognition
    
    Returns:
    - Raw JSON with detection and recognition results
    """
    results = await recognize_monument(background_tasks, image, location)
    json_results = monument_pipeline.to_json(results)
    return JSONResponse(content=json.loads(json_results))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)