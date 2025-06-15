"""
Pydantic schemas for emotion classification API.

This module defines the request and response models for the emotion
classification endpoints.
"""

from typing import List, Dict, Any
from pydantic import BaseModel, Field, validator, ConfigDict


class EmotionPrediction(BaseModel):
    """Schema for a single emotion prediction."""
    
    label: str = Field(..., description="The emotion label")
    score: float = Field(..., description="Confidence score for the emotion", ge=0.0, le=1.0)


class TextClassificationRequest(BaseModel):
    """Schema for single text classification request."""
    
    text: str = Field(..., description="The text to classify", min_length=1, max_length=10000)
    
    @validator('text')
    def validate_text(cls, v):
        """Validate that text is not empty after stripping whitespace."""
        if not v.strip():
            raise ValueError('Text cannot be empty or contain only whitespace')
        return v.strip()


class TextClassificationResponse(BaseModel):
    """Schema for single text classification response."""
    
    text: str = Field(..., description="The input text that was classified")
    predictions: List[EmotionPrediction] = Field(..., description="List of emotion predictions")
    top_emotion: str = Field(..., description="The most likely emotion")
    confidence: float = Field(..., description="Confidence score of the top emotion")


class BatchTextClassificationRequest(BaseModel):
    """Schema for batch text classification request."""
    
    texts: List[str] = Field(..., description="List of texts to classify", min_items=1, max_items=100)
    
    @validator('texts')
    def validate_texts(cls, v):
        """Validate that all texts are not empty after stripping whitespace."""
        cleaned_texts = []
        for text in v:
            if not text.strip():
                raise ValueError('All texts must contain non-whitespace characters')
            cleaned_texts.append(text.strip())
        return cleaned_texts


class BatchTextClassificationResponse(BaseModel):
    """Schema for batch text classification response."""
    
    results: List[TextClassificationResponse] = Field(..., description="List of classification results")
    total_processed: int = Field(..., description="Total number of texts processed")


class TopEmotionRequest(BaseModel):
    """Schema for top emotion only request."""
    
    text: str = Field(..., description="The text to classify", min_length=1, max_length=10000)
    
    @validator('text')
    def validate_text(cls, v):
        """Validate that text is not empty after stripping whitespace."""
        if not v.strip():
            raise ValueError('Text cannot be empty or contain only whitespace')
        return v.strip()


class TopEmotionResponse(BaseModel):
    """Schema for top emotion only response."""
    
    text: str = Field(..., description="The input text that was classified")
    emotion: str = Field(..., description="The most likely emotion")
    confidence: float = Field(..., description="Confidence score", ge=0.0, le=1.0)


class ModelInfoResponse(BaseModel):
    """Schema for model information response."""
    
    model_config = ConfigDict(protected_namespaces=())
    
    model_name: str = Field(..., description="Name of the loaded model")
    is_loaded: str = Field(..., description="Whether the model is loaded")
    device: str = Field(..., description="Device the model is running on")
    status: str = Field(..., description="Current status of the model")


class HealthCheckResponse(BaseModel):
    """Schema for health check response."""
    
    model_config = ConfigDict(protected_namespaces=())
    
    status: str = Field(..., description="Health status")
    model_loaded: bool = Field(..., description="Whether the model is loaded")
    timestamp: str = Field(..., description="Current timestamp")


class ErrorResponse(BaseModel):
    """Schema for error responses."""
    
    error: str = Field(..., description="Error message")
    detail: str = Field(..., description="Detailed error information")
    status_code: int = Field(..., description="HTTP status code")


class AvailableEmotionsResponse(BaseModel):
    """Schema for available emotions response."""
    
    emotions: List[str] = Field(..., description="List of all available emotion labels")
    total_emotions: int = Field(..., description="Total number of available emotions")
    model_name: str = Field(..., description="Name of the model providing these emotions") 