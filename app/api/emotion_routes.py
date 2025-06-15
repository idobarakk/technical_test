"""
Emotion classification API routes.

This module defines all the API endpoints for emotion classification,
including single text, batch processing, and utility endpoints.
"""

import logging
from datetime import datetime
from typing import List
from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.responses import JSONResponse

from app.models.emotion_classifier import EmotionClassifier
from app.schemas.emotion_schemas import (
    TextClassificationRequest,
    TextClassificationResponse,
    # BatchTextClassificationRequest,
    # BatchTextClassificationResponse,
    TopEmotionRequest,
    TopEmotionResponse,
    ModelInfoResponse,
    HealthCheckResponse,
    ErrorResponse,
    EmotionPrediction,
    AvailableEmotionsResponse
)

# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/v1", tags=["emotion-classification"])

# Global classifier instance
classifier = EmotionClassifier()


def get_classifier() -> EmotionClassifier:
    """
    Dependency to get the classifier instance.
    
    Returns:
        The emotion classifier instance.
        
    Raises:
        HTTPException: If the model is not loaded.
    """
    if not classifier.is_loaded:
        logger.error("Model is not loaded")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded. Please try again later."
        )
    return classifier


@router.post(
    "/classify",
    response_model=TextClassificationResponse,
    summary="Classify emotion in text",
    description="Classify the emotion in a single text and return all emotions with confidence scores."
)
async def classify_emotion(
    request: TextClassificationRequest,
    emotion_classifier: EmotionClassifier = Depends(get_classifier)
) -> TextClassificationResponse:
    """
    Classify emotion in a single text.
    
    Args:
        request: The text classification request.
        emotion_classifier: The emotion classifier instance.
        
    Returns:
        Classification results with all emotions and confidence scores.
        
    Raises:
        HTTPException: If classification fails.
    """
    try:
        logger.info(f"Classifying text: {request.text[:50]}...")
        
        predictions = emotion_classifier.predict(request.text)
        
        emotion_predictions = [
            EmotionPrediction(label=pred['label'], score=pred['score'])
            for pred in predictions
        ]
        
        response = TextClassificationResponse(
            text=request.text,
            predictions=emotion_predictions,
            top_emotion=predictions[0]['label'],
            confidence=predictions[0]['score']
        )
        
        logger.info(f"Classification successful. Top emotion: {response.top_emotion}")
        return response
        
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Classification failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during classification"
        )


@router.post(
    "/classify/top",
    response_model=TopEmotionResponse,
    summary="Get top emotion only",
    description="Classify the emotion in text and return only the most likely emotion."
)
async def classify_top_emotion(
    request: TopEmotionRequest,
    emotion_classifier: EmotionClassifier = Depends(get_classifier)
) -> TopEmotionResponse:
    """
    Get only the top emotion for a text.
    
    Args:
        request: The text classification request.
        emotion_classifier: The emotion classifier instance.
        
    Returns:
        The top emotion and confidence score.
    """
    try:
        logger.info(f"Getting top emotion for text: {request.text[:50]}...")
        
        top_prediction = emotion_classifier.predict_top_emotion(request.text)
        
        response = TopEmotionResponse(
            text=request.text,
            emotion=top_prediction['label'],
            confidence=top_prediction['score']
        )
        
        logger.info(f"Top emotion: {response.emotion}")
        return response
        
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Classification failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during classification"
        )


# @router.post(
#     "/classify/batch",
#     response_model=BatchTextClassificationResponse,
#     summary="Classify emotions in multiple texts",
#     description="Classify emotions in a batch of texts (up to 100 texts)."
# )
# async def classify_batch_emotions(
#     request: BatchTextClassificationRequest,
#     emotion_classifier: EmotionClassifier = Depends(get_classifier)
# ) -> BatchTextClassificationResponse:
#     """
#     Classify emotions in a batch of texts.
#     
#     Args:
#         request: The batch text classification request.
#         emotion_classifier: The emotion classifier instance.
#         
#     Returns:
#         Batch classification results.
#     """
#     try:
#         logger.info(f"Processing batch of {len(request.texts)} texts")
#         
#         batch_predictions = emotion_classifier.predict_batch(request.texts)
#         
#         results = []
#         for i, (text, predictions) in enumerate(zip(request.texts, batch_predictions)):
#             if predictions:  # If prediction was successful
#                 emotion_predictions = [
#                     EmotionPrediction(label=pred['label'], score=pred['score'])
#                     for pred in predictions
#                 ]
#                 
#                 result = TextClassificationResponse(
#                     text=text,
#                     predictions=emotion_predictions,
#                     top_emotion=predictions[0]['label'],
#                     confidence=predictions[0]['score']
#                 )
#                 results.append(result)
#             else:
#                 logger.warning(f"Failed to process text at index {i}")
#         
#         response = BatchTextClassificationResponse(
#             results=results,
#             total_processed=len(results)
#         )
#         
#         logger.info(f"Batch processing completed. {len(results)}/{len(request.texts)} successful")
#         return response
#         
#     except ValueError as e:
#         logger.error(f"Validation error: {str(e)}")
#         raise HTTPException(
#             status_code=status.HTTP_400_BAD_REQUEST,
#             detail=str(e)
#         )
#     except Exception as e:
#         logger.error(f"Batch classification failed: {str(e)}")
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail="Internal server error during batch classification"
#         )


@router.get(
    "/model/info",
    response_model=ModelInfoResponse,
    summary="Get model information",
    description="Get information about the loaded emotion classification model."
)
async def get_model_info() -> ModelInfoResponse:
    """
    Get information about the loaded model.
    
    Returns:
        Model information including name, status, and device.
    """
    try:
        model_info = classifier.get_model_info()
        
        response = ModelInfoResponse(
            model_name=model_info['model_name'],
            is_loaded=model_info['is_loaded'],
            device=model_info['device'],
            status="ready" if classifier.is_loaded else "loading"
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Failed to get model info: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve model information"
        )


@router.get(
    "/emotions",
    response_model=AvailableEmotionsResponse,
    summary="Get available emotions",
    description="Get a list of all emotions that can be classified by the model."
)
async def get_available_emotions(
    emotion_classifier: EmotionClassifier = Depends(get_classifier)
) -> AvailableEmotionsResponse:
    """
    Get all available emotions that can be classified.
    
    Args:
        emotion_classifier: The emotion classifier instance.
        
    Returns:
        List of all available emotion labels.
        
    Raises:
        HTTPException: If unable to retrieve emotions.
    """
    try:
        logger.info("Retrieving available emotions...")
        
        emotions = emotion_classifier.get_available_emotions()
        
        response = AvailableEmotionsResponse(
            emotions=emotions,
            total_emotions=len(emotions),
            model_name=emotion_classifier.model_name
        )
        
        logger.info(f"Retrieved {len(emotions)} available emotions")
        return response
        
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Failed to get available emotions: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve available emotions"
        )


@router.get(
    "/health",
    response_model=HealthCheckResponse,
    summary="Health check",
    description="Check the health status of the emotion classification service."
)
async def health_check() -> HealthCheckResponse:
    """
    Perform a health check of the service.
    
    Returns:
        Health status information.
    """
    try:
        current_time = datetime.now().isoformat()
        
        response = HealthCheckResponse(
            status="healthy" if classifier.is_loaded else "loading",
            model_loaded=classifier.is_loaded,
            timestamp=current_time
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return HealthCheckResponse(
            status="unhealthy",
            model_loaded=False,
            timestamp=datetime.now().isoformat()
        )


# @router.post(
#     "/model/reload",
#     summary="Reload model",
#     description="Reload the emotion classification model."
# )
# async def reload_model() -> dict:
#     """
#     Reload the emotion classification model.
#     
#     Returns:
#         Status message about the reload operation.
#     """
#     try:
#         logger.info("Reloading model...")
#         classifier.load_model()
#         logger.info("Model reloaded successfully")
#         
#         return {
#             "message": "Model reloaded successfully",
#             "status": "success",
#             "timestamp": datetime.now().isoformat()
#         }
#         
#     except Exception as e:
#         logger.error(f"Model reload failed: {str(e)}")
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail="Failed to reload model"
#         ) 