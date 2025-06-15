"""
Emotion Classification API - Main Application

A FastAPI-based web service for emotion classification using HuggingFace transformers.
This application provides endpoints for classifying emotions in text with comprehensive
API documentation via Swagger UI.
"""

import logging
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.exception_handlers import http_exception_handler
import uvicorn

from app.api.emotion_routes import router as emotion_router, classifier
from app.schemas.emotion_schemas import ErrorResponse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('emotion_api.log')
    ]
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    
    Handles startup and shutdown events for the FastAPI application.
    """
    # Startup
    logger.info("Starting Emotion Classification API...")
    
    # Load the model on startup
    try:
        logger.info("Loading emotion classification model...")
        classifier.load_model()
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model on startup: {str(e)}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Emotion Classification API...")


# Create FastAPI application
app = FastAPI(
    title="Emotion Classification API",
    description="""
    ## Emotion Classification API

    A powerful and efficient API for classifying emotions in text using state-of-the-art
    HuggingFace transformer models.

    ### Features:
    - **Single Text Classification**: Classify emotions in individual texts
    - **Top Emotion Only**: Get just the most likely emotion for efficiency
    - **Model Information**: Access details about the loaded model
    - **Health Monitoring**: Check service health and model status
    - **Comprehensive Error Handling**: Detailed error responses with proper HTTP status codes

    ### Supported Emotions:
    The model can classify the following emotions:
    - Anger, Cheeky, Confuse, Curious, Disgust
    - Empathetic, Energetic, Fear, Grumpy, Guilty  
    - Impatient, Joy, Love, Neutral, Sadness
    - Serious, Surprise, Suspicious, Think, Whiny
    
    *Total: 20 different emotions*

    ### Model Information:
    - **Model**: jitesh/emotion-english
    - **Framework**: HuggingFace Transformers
    - **Type**: BERT-based sequence classification
    """,
    version="1.0.0",
    contact={
        "name": "Emotion Classification API Support",
        "email": "support@emotion-api.com",
    },
    license_info={
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT",
    },
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Custom exception handlers
@app.exception_handler(HTTPException)
async def custom_http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """
    Custom HTTP exception handler.
    
    Args:
        request: The HTTP request that caused the exception.
        exc: The HTTP exception that was raised.
        
    Returns:
        A JSON response with error details.
    """
    logger.error(f"HTTP Exception: {exc.status_code} - {exc.detail}")
    
    error_response = ErrorResponse(
        error=exc.detail,
        detail=f"HTTP {exc.status_code} Error",
        status_code=exc.status_code
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content=error_response.dict()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    General exception handler for unexpected errors.
    
    Args:
        request: The HTTP request that caused the exception.
        exc: The exception that was raised.
        
    Returns:
        A JSON response with error details.
    """
    logger.error(f"Unexpected error: {str(exc)}", exc_info=True)
    
    error_response = ErrorResponse(
        error="Internal server error",
        detail="An unexpected error occurred. Please try again later.",
        status_code=500
    )
    
    return JSONResponse(
        status_code=500,
        content=error_response.dict()
    )


# Include routers
app.include_router(emotion_router)


@app.get("/", include_in_schema=False)
async def root():
    """
    Root endpoint that redirects to the API documentation.
    
    Returns:
        A redirect response to the Swagger UI documentation.
    """
    return RedirectResponse(url="/docs")


# @app.get("/info", tags=["general"])
# async def get_api_info() -> Dict[str, Any]:
#     """
#     Get general information about the API.
#     
#     Returns:
#         API information including version, title, and description.
#     """
#     return {
#         "title": app.title,
#         "description": "Emotion Classification API using HuggingFace transformers",
#         "version": app.version,
#         "docs_url": "/docs",
#         "redoc_url": "/redoc",
#         "openapi_url": "/openapi.json"
#     }


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    ) 