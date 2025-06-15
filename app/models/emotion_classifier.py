"""
Emotion classification model wrapper using HuggingFace transformers.

This module provides a clean interface for emotion classification using the
jitesh/emotion-english model.
"""

import logging
from typing import Dict, List, Optional, Any
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    pipeline,
    Pipeline
)
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmotionClassifier:
    """
    A wrapper class for emotion classification using HuggingFace transformers.
    
    This class provides a clean interface for loading and using the emotion
    classification model with proper error handling and logging.
    """
    
    def __init__(self, model_name: str = "jitesh/emotion-english") -> None:
        """
        Initialize the emotion classifier.
        
        Args:
            model_name: The name of the HuggingFace model to use for classification.
        """
        self.model_name = model_name
        self._model: Optional[AutoModelForSequenceClassification] = None
        self._tokenizer: Optional[AutoTokenizer] = None
        self._classifier: Optional[Pipeline] = None
        self._is_loaded = False
        
        logger.info(f"Initializing EmotionClassifier with model: {model_name}")
    
    def load_model(self) -> None:
        """
        Load the model, tokenizer, and create the classification pipeline.
        
        Raises:
            Exception: If model loading fails.
        """
        try:
            logger.info("Loading model and tokenizer...")
            
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            
            self._classifier = pipeline(
                "text-classification",
                model=self._model,
                tokenizer=self._tokenizer,
                top_k=None  # Return all scores (replaces deprecated return_all_scores=True)
            )
            
            self._is_loaded = True
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise
    
    def predict(self, text: str) -> List[Dict[str, Any]]:
        """
        Classify the emotion of the given text.
        
        Args:
            text: The input text to classify.
            
        Returns:
            A list of dictionaries containing emotion labels and confidence scores.
            
        Raises:
            ValueError: If the model is not loaded or text is empty.
            Exception: If prediction fails.
        """
        if not self._is_loaded:
            raise ValueError("Model is not loaded. Call load_model() first.")
        
        if not text or not text.strip():
            raise ValueError("Input text cannot be empty.")
        
        try:
            logger.debug(f"Classifying text: {text[:50]}...")
            
            predictions = self._classifier(text.strip())
            
            # Handle the format returned by pipeline with top_k=None
            # Pipeline returns a list of predictions for the input text
            if isinstance(predictions, list) and len(predictions) > 0:
                # If predictions is a list of lists (batch format), take the first element
                if isinstance(predictions[0], list):
                    predictions = predictions[0]
            
            # Sort by score in descending order
            sorted_predictions = sorted(predictions, key=lambda x: x['score'], reverse=True)
            
            logger.debug(f"Predictions: {sorted_predictions[:3]}...")  # Log only top 3 for brevity
            return sorted_predictions
            
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise
    
    def predict_top_emotion(self, text: str) -> Dict[str, Any]:
        """
        Get the top emotion prediction for the given text.
        
        Args:
            text: The input text to classify.
            
        Returns:
            A dictionary containing the top emotion label and confidence score.
        """
        predictions = self.predict(text)
        return predictions[0] if predictions else {}
    
    def predict_batch(self, texts: List[str]) -> List[List[Dict[str, Any]]]:
        """
        Classify emotions for a batch of texts.
        
        Args:
            texts: List of input texts to classify.
            
        Returns:
            A list of prediction lists, one for each input text.
        """
        if not self._is_loaded:
            raise ValueError("Model is not loaded. Call load_model() first.")
        
        logger.info(f"Processing batch of {len(texts)} texts")
        
        results = []
        for text in texts:
            try:
                predictions = self.predict(text)
                results.append(predictions)
            except Exception as e:
                logger.error(f"Failed to process text '{text[:30]}...': {str(e)}")
                results.append([])
        
        return results
    
    @property
    def is_loaded(self) -> bool:
        """Check if the model is loaded and ready for predictions."""
        return self._is_loaded
    
    def get_model_info(self) -> Dict[str, str]:
        """
        Get information about the loaded model.
        
        Returns:
            A dictionary containing model information.
        """
        return {
            "model_name": self.model_name,
            "is_loaded": str(self._is_loaded),
            "device": str(next(self._model.parameters()).device) if self._model else "not_loaded"
        }
    
    def get_available_emotions(self) -> List[str]:
        """
        Get list of all available emotion labels that the model can predict.
        
        Returns:
            A list of emotion label strings.
            
        Raises:
            ValueError: If the model is not loaded.
        """
        if not self._is_loaded:
            raise ValueError("Model is not loaded. Call load_model() first.")
        
        try:
            # Get the label mappings from the model configuration
            if hasattr(self._model.config, 'id2label'):
                emotions = list(self._model.config.id2label.values())
                logger.debug(f"Found {len(emotions)} emotions from model config")
                return sorted(emotions)
            else:
                # Fallback to known emotions for this specific model
                logger.warning("Could not get emotions from model config, using fallback list")
                return sorted([
                    "anger", "disgust", "fear", "joy", "neutral", "sadness",
                    "surprise", "love", "optimism", "pessimism", "annoyance",
                    "grief", "disapproval", "realization", "nervousness",
                    "approval", "curiosity", "admiration", "excitement",
                    "gratitude", "pride", "amusement", "desire", "caring",
                    "confusion", "embarrassment", "remorse", "disappointment"
                ])
                
        except Exception as e:
            logger.error(f"Failed to get available emotions: {str(e)}")
            raise 