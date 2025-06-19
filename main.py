"""
Emotion Classification API - VADER Edition
A FastAPI-based web service for emotion classification using VADER sentiment analysis.
Ultra-lightweight containerized application for text emotion classification.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import logging
import os
import re
from contextlib import asynccontextmanager
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global variables for model initialization
is_loaded = False
vader_analyzer = None

# Emotion keywords mapping for enhanced emotion detection
EMOTION_KEYWORDS = {
    'happy': ['happy', 'joy', 'joyful', 'cheerful', 'delighted', 'pleased', 'glad', 'excited', 'elated', 'euphoric', 'blissful', 'content', 'satisfied', 'wonderful', 'amazing', 'fantastic', 'great', 'excellent', 'perfect', 'awesome', 'brilliant', 'marvelous', 'terrific', 'superb', 'magnificent', 'love', 'adore', 'enjoy'],
    'sad': ['sad', 'sadness', 'unhappy', 'depressed', 'melancholy', 'sorrowful', 'grief', 'heartbroken', 'devastated', 'miserable', 'gloomy', 'down', 'blue', 'dejected', 'despondent', 'tragic', 'mourning', 'regret', 'disappointed', 'discouraged', 'hopeless', 'despair'],
    'angry': ['angry', 'anger', 'mad', 'furious', 'rage', 'outraged', 'irritated', 'annoyed', 'frustrated', 'enraged', 'livid', 'incensed', 'irate', 'infuriated', 'resentful', 'hostile', 'agitated', 'upset', 'bothered', 'aggravated', 'hate', 'disgusted', 'revolted'],
    'fear': ['fear', 'afraid', 'scared', 'terrified', 'frightened', 'anxious', 'worried', 'nervous', 'panic', 'terror', 'dread', 'alarmed', 'apprehensive', 'uneasy', 'concerned', 'stressed', 'tense', 'paranoid', 'horrified', 'petrified'],
    'surprise': ['surprise', 'surprised', 'shocking', 'unexpected', 'astonished', 'amazed', 'stunned', 'bewildered', 'confused', 'puzzled', 'perplexed', 'baffled', 'wow', 'omg', 'unbelievable', 'incredible', 'remarkable', 'extraordinary']
}

class TextInput(BaseModel):
    """Input model for text classification requests."""
    text: str

class TopEmotionRequest(BaseModel):
    """Input model for top emotion requests."""
    text: str

class EmotionPrediction(BaseModel):
    """Model for individual emotion prediction."""
    emotion: str
    score: float

class EmotionResponse(BaseModel):
    """Response model for complete emotion analysis."""
    text: str
    predictions: List[EmotionPrediction]
    top_emotion: str
    confidence: float

class TopEmotionResponse(BaseModel):
    """Response model for top emotion only."""
    text: str
    emotion: str
    confidence: float

class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    model_loaded: bool
    message: str

class AvailableEmotionsResponse(BaseModel):
    """Response model for available emotions list."""
    emotions: List[str]
    total_count: int

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize VADER on startup."""
    global is_loaded, vader_analyzer
    
    try:
        logger.info("Initializing emotion classification system...")
        
        # Initialize VADER
        vader_analyzer = SentimentIntensityAnalyzer()
        
        # Test VADER analyzer
        test_text = "I am happy"
        vader_scores = vader_analyzer.polarity_scores(test_text)
        
        logger.info("VADER analyzer initialized successfully")
        
        is_loaded = True
        logger.info("Emotion classification system loaded successfully")
        
    except Exception as e:
        logger.error(f"Error initializing emotion classification system: {e}")
        raise e
    
    yield
    
    # Cleanup
    logger.info("Application shutdown complete")

# Initialize FastAPI app
app = FastAPI(
    title="Emotion Classification API - VADER Edition",
    description="""
     Ultra-Lightweight Text Emotion Classification API

    # Available Emotions
    The system can detect the following emotional categories:
    
    Primary Emotions:
    - Happy
    - Sad 
    - Angry 
    - Fear 
    - Surprise  

    # API Endpoints
    - GET / - Basic API information and health status
    - POST /classify - Complete emotion analysis with all emotion confidence scores
    - POST /classify/top - Returns only the highest confidence emotion
    - GET /health - Service health check and model status
    - GET /emotions - List of all available emotion categories

    # Input Requirements
    - Content-Type: `application/json`
    - Text Field: Non-empty string (max 50 characters)
    - Language: Best performance with English text

    # Response Format
    All successful responses return JSON with appropriate data structure.
    

     Success Codes
    - 200 OK - Request processed successfully


     Client Error Codes 
    - 400 Bad Request - Invalid input provided
    - 413 Request Entity Too Large - Input exceeds size limits  
    - 422 Unprocessable Entity - Validation error

     Server Error Codes (5xx) 
    - 500 Internal Server Error- Unexpected server errors
    - 503 Service Unavailable - Service temporarily unavailable

    # Error Response Structure
    All error responses follow this format:
    ```json
    {
        "detail": "Human-readable error description"
    }
    ```


    # Citation
    VADER: Hutto, C.J. & Gilbert, E.E. (2014). VADER: A Parsimonious Rule-based Model for 
    Sentiment Analysis of Social Media Text. Eighth International Conference on Weblogs and Social Media (ICWSM-14).
    """,
    version="4.0.0",
    lifespan=lifespan
)

def detect_emotions_from_keywords(text: str) -> Dict[str, float]:
    """
    Detect emotions based on keyword matching.
    
    Args:
        text (str): Input text to analyze
        
    Returns:
        Dict[str, float]: Emotion scores based on keyword frequency
    """
    text_lower = text.lower()
    emotion_scores = {}
    
    for emotion, keywords in EMOTION_KEYWORDS.items():
        score = 0
        for keyword in keywords:
            # Count occurrences with word boundaries to avoid partial matches
            pattern = r'\b' + re.escape(keyword) + r'\b'
            matches = len(re.findall(pattern, text_lower))
            score += matches
        
        if score > 0:
            # Normalize by text length to get relative frequency
            normalized_score = min(score / len(text_lower.split()) * 10, 1.0)
            emotion_scores[emotion] = normalized_score
    
    return emotion_scores

def predict_emotion(text: str) -> List[Dict[str, Any]]:
    """
    Predict emotions for given text using keyword-based emotion detection enhanced with VADER sentiment.
    
    Args:
        text (str): Input text to analyze
        
    Returns:
        List[Dict[str, Any]]: List of ALL emotion predictions with scores (always returns all 5 emotions)
        
    Raises:
        HTTPException: For various error conditions
    """
    if not is_loaded or vader_analyzer is None:
        logger.error("Emotion analysis system is not initialized")
        raise HTTPException(status_code=503, detail="Emotion classification system is not loaded")
    
    if not text or not text.strip():
        logger.warning("Empty text input received")
        raise HTTPException(status_code=400, detail="Input text cannot be empty")
    
    # Check text length limit
    text_cleaned = text.strip()
    if len(text_cleaned) > 30:
        logger.warning(f"Text length exceeded limit: {len(text_cleaned)} characters")
        raise HTTPException(status_code=413, detail="Text exceeds 50 character limit")
    
    try:
        logger.info(f"Analyzing text: '{text_cleaned[:100]}{'...' if len(text_cleaned) > 100 else ''}'")
        
        # Initialize all emotions with 0.0 score
        all_emotions = ['happy', 'sad', 'angry', 'fear', 'surprise']
        emotion_scores_dict = {emotion: 0.0 for emotion in all_emotions}
        
        # Keyword-based emotion detection
        detected_emotions = detect_emotions_from_keywords(text_cleaned)
        
        # Update scores for detected emotions
        for emotion, score in detected_emotions.items():
            if emotion in emotion_scores_dict:
                emotion_scores_dict[emotion] = score
        
        # If no specific emotions detected, enhance based on VADER scores
        if not detected_emotions:
            # Use VADER to help determine likely emotions when keywords aren't found
            vader_scores = vader_analyzer.polarity_scores(text_cleaned)
            compound_score = vader_scores['compound']
        
            if compound_score > 0.3:
                # Strong positive sentiment - likely happy
                emotion_scores_dict['happy'] = compound_score * 0.8
            elif compound_score < -0.3:
                # Strong negative sentiment - determine if sad or angry based on intensity
                if vader_scores.get('neg', 0) > 0.6:
                    # High negative intensity - likely angry
                    emotion_scores_dict['angry'] = abs(compound_score) * 0.7
                else:
                    # Moderate negative - likely sad
                    emotion_scores_dict['sad'] = abs(compound_score) * 0.7
            else:
                # Neutral or weak sentiment - default to very low happy
                emotion_scores_dict['happy'] = 0.1
        
        # Convert to list format and sort by score (descending)
        predictions = [
            {'emotion': emotion, 'score': score}
            for emotion, score in emotion_scores_dict.items()
        ]
        
        # Sort by score in descending order
        predictions_sorted = sorted(predictions, key=lambda x: x['score'], reverse=True)
        
        logger.info(f"Emotion classification successful. Returning all {len(predictions_sorted)} emotions with scores")
        return predictions_sorted
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Emotion classification error: {str(e)}")

@app.get("/")
async def root():
    """Root endpoint providing API information."""
    return {
        "message": "Ultra-Lightweight Emotion Classification API is running",
        "engine": "VADER + Keyword-based emotion detection",
        "version": "4.0.0",
        "status": "healthy" if is_loaded else "initializing",
        "emotions": ["happy", "sad", "angry", "fear", "surprise"],
        "features": [
            "Ultra-lightweight dependencies (~2MB)",
            "Sub-second startup time",
            "Sub-millisecond response time",
            "Keyword-based emotion detection enhanced with VADER",
            "Social media text optimized",
            "5 core emotions: happy, sad, angry, fear, surprise",
            "No complex dependencies"
        ],
        "docs": "/docs"
    }

@app.post("/classify", response_model=EmotionResponse)
async def classify_emotion(request: TextInput) -> EmotionResponse:
    """
    Classify emotions in text and return all detected emotions with confidence scores sorted by confidence (descending).
    
    Args:
        request (TextInput): Text input for emotion classification
        
    Returns:
        EmotionResponse: Complete emotion analysis with all predictions sorted by confidence
    """
    try:
        logger.info(f"Processing classification request for text: {request.text[:50]}...")
        
        # Get predictions sorted by confidence in descending order
        predictions = predict_emotion(request.text)
        
        # Convert to response format
        emotion_predictions = [
            EmotionPrediction(emotion=pred['emotion'], score=round(pred['score'], 4))
            for pred in predictions
        ]
        
        # Top emotion is the first (highest confidence) prediction
        top_emotion = predictions[0]['emotion']
        top_confidence = round(predictions[0]['score'], 4)
        
        response = EmotionResponse(
            text=request.text,
            predictions=emotion_predictions,
            top_emotion=top_emotion,
            confidence=top_confidence
        )
        
        logger.info(f"Classification successful. Detected {len(emotion_predictions)} emotions. Top: {top_emotion} (confidence: {top_confidence})")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Classification failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error during classification")

@app.post("/classify/top", response_model=TopEmotionResponse)
async def classify_top_emotion(request: TopEmotionRequest) -> TopEmotionResponse:
    """
    Get only the top emotion (highest confidence) for a text.
    
    Args:
        request (TopEmotionRequest): Text input for top emotion classification
        
    Returns:
        TopEmotionResponse: Top emotion result with highest confidence score
    """
    try:
        logger.info(f"Processing top emotion request for text: {request.text[:50]}...")
        
        # Get predictions (already sorted by confidence descending)
        predictions = predict_emotion(request.text)
        
        # Get the top prediction (first item)
        top_prediction = predictions[0]
        
        response = TopEmotionResponse(
            text=request.text,
            emotion=top_prediction['emotion'],
            confidence=round(top_prediction['score'], 4)
        )
        
        logger.info(f"Top emotion classification: {response.emotion} (confidence: {response.confidence})")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Top emotion classification failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error during classification")

@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """
    Check the health status of the emotion classification service.
    
    Returns:
        HealthResponse: Service health information
    """
    return HealthResponse(
        status="healthy" if is_loaded else "unhealthy",
        model_loaded=is_loaded,
        message="VADER emotion classification system is ready" if is_loaded else "System not initialized"
    )

@app.get("/emotions", response_model=AvailableEmotionsResponse)
async def get_available_emotions() -> AvailableEmotionsResponse:
    """
    Get list of available emotions that can be detected.
    
    Returns:
        AvailableEmotionsResponse: Simple list of available emotions (emotions only, no sentiment)
    """
    if not is_loaded:
        logger.error("Service not available - emotion analyzer not initialized")
        raise HTTPException(status_code=503, detail="Emotion classification system is not loaded")
    
    try:
        # Get only emotion categories (no sentiment analysis categories)
        available_emotions = list(EMOTION_KEYWORDS.keys())  # ['happy', 'sad', 'angry', 'fear', 'surprise']
        
        logger.info(f"Returning {len(available_emotions)} available emotions")
        available_emotions = ['happy', 'sad', 'angry', 'fear', 'surprise',"neutral","joy","anxiety"]
        return AvailableEmotionsResponse(
            emotions=available_emotions,
            total_count=5
        )
        
    except Exception as e:
        logger.error(f"Failed to get emotions list: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve emotions information")

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Ultra-Lightweight Emotion Classification API...")
    uvicorn.run(app, host="0.0.0.0", port=8000) 