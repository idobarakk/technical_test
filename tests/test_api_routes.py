"""
Integration tests for the emotion classification API routes.

This module contains comprehensive tests for all API endpoints including
success cases, error handling, and edge cases.
"""

import pytest
from unittest.mock import Mock, patch
from fastapi.testclient import TestClient
from typing import Dict, Any

from main import app
from app.models.emotion_classifier import EmotionClassifier


# Test client
client = TestClient(app=app)


class TestEmotionRoutes:
    """Test cases for emotion classification API routes."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.base_url = "/api/v1"
    
    @patch('app.api.emotion_routes.classifier')
    def test_classify_emotion_success(self, mock_classifier):
        """Test successful single text classification."""
        # Setup mock
        mock_classifier.is_loaded = True
        mock_predictions = [
            {'label': 'joy', 'score': 0.85},
            {'label': 'sadness', 'score': 0.15}
        ]
        mock_classifier.predict.return_value = mock_predictions
        
        # Test request
        response = client.post(
            f"{self.base_url}/classify",
            json={"text": "I am very happy today!"}
        )
        
        # Assertions
        assert response.status_code == 200
        data = response.json()
        assert data["text"] == "I am very happy today!"
        assert data["top_emotion"] == "joy"
        assert data["confidence"] == 0.85
        assert len(data["predictions"]) == 2
        assert data["predictions"][0]["label"] == "joy"
        assert data["predictions"][0]["score"] == 0.85
    
    @patch('app.api.emotion_routes.classifier')
    def test_classify_emotion_model_not_loaded(self, mock_classifier):
        """Test classification when model is not loaded."""
        # Setup mock
        mock_classifier.is_loaded = False
        
        # Test request
        response = client.post(
            f"{self.base_url}/classify",
            json={"text": "I am happy!"}
        )
        
        # Assertions
        assert response.status_code == 503
        data = response.json()
        assert "Model is not loaded" in data["error"]
    
    def test_classify_emotion_empty_text(self):
        """Test classification with empty text."""
        response = client.post(
            f"{self.base_url}/classify",
            json={"text": ""}
        )
        
        # Assertions
        assert response.status_code == 422  # Validation error
    
    def test_classify_emotion_whitespace_only(self):
        """Test classification with whitespace-only text."""
        response = client.post(
            f"{self.base_url}/classify",
            json={"text": "   "}
        )
        
        # Assertions
        assert response.status_code == 422  # Validation error
    
    def test_classify_emotion_missing_text(self):
        """Test classification with missing text field."""
        response = client.post(
            f"{self.base_url}/classify",
            json={}
        )
        
        # Assertions
        assert response.status_code == 422  # Validation error
    
    def test_classify_emotion_text_too_long(self):
        """Test classification with text exceeding maximum length."""
        long_text = "a" * 10001  # Exceeds max length of 10000
        response = client.post(
            f"{self.base_url}/classify",
            json={"text": long_text}
        )
        
        # Assertions
        assert response.status_code == 422  # Validation error
    
    @patch('app.api.emotion_routes.classifier')
    def test_classify_top_emotion_success(self, mock_classifier):
        """Test successful top emotion classification."""
        # Setup mock
        mock_classifier.is_loaded = True
        mock_prediction = {'label': 'joy', 'score': 0.85}
        mock_classifier.predict_top_emotion.return_value = mock_prediction
        
        # Test request
        response = client.post(
            f"{self.base_url}/classify/top",
            json={"text": "I am very happy today!"}
        )
        
        # Assertions
        assert response.status_code == 200
        data = response.json()
        assert data["text"] == "I am very happy today!"
        assert data["emotion"] == "joy"
        assert data["confidence"] == 0.85
    
    @patch('app.api.emotion_routes.classifier')
    def test_classify_batch_success(self, mock_classifier):
        """Test successful batch classification."""
        # Setup mock
        mock_classifier.is_loaded = True
        mock_batch_predictions = [
            [{'label': 'joy', 'score': 0.8}],
            [{'label': 'sadness', 'score': 0.7}]
        ]
        mock_classifier.predict_batch.return_value = mock_batch_predictions
        
        # Test request
        response = client.post(
            f"{self.base_url}/classify/batch",
            json={"texts": ["I am happy!", "I am sad."]}
        )
        
        # Assertions
        assert response.status_code == 200
        data = response.json()
        assert data["total_processed"] == 2
        assert len(data["results"]) == 2
        assert data["results"][0]["top_emotion"] == "joy"
        assert data["results"][1]["top_emotion"] == "sadness"
    
    def test_classify_batch_empty_list(self):
        """Test batch classification with empty list."""
        response = client.post(
            f"{self.base_url}/classify/batch",
            json={"texts": []}
        )
        
        # Assertions
        assert response.status_code == 422  # Validation error
    
    def test_classify_batch_too_many_texts(self):
        """Test batch classification with too many texts."""
        texts = ["text"] * 101  # Exceeds max of 100
        response = client.post(
            f"{self.base_url}/classify/batch",
            json={"texts": texts}
        )
        
        # Assertions
        assert response.status_code == 422  # Validation error
    
    def test_classify_batch_empty_text_in_list(self):
        """Test batch classification with empty text in list."""
        response = client.post(
            f"{self.base_url}/classify/batch",
            json={"texts": ["valid text", "", "another valid text"]}
        )
        
        # Assertions
        assert response.status_code == 422  # Validation error
    
    @patch('app.api.emotion_routes.classifier')
    def test_get_model_info_success(self, mock_classifier):
        """Test successful model info retrieval."""
        # Setup mock
        mock_info = {
            'model_name': 'jitesh/emotion-english',
            'is_loaded': 'True',
            'device': 'cpu'
        }
        mock_classifier.get_model_info.return_value = mock_info
        mock_classifier.is_loaded = True
        
        # Test request
        response = client.get(f"{self.base_url}/model/info")
        
        # Assertions
        assert response.status_code == 200
        data = response.json()
        assert data["model_name"] == "jitesh/emotion-english"
        assert data["is_loaded"] == "True"
        assert data["device"] == "cpu"
        assert data["status"] == "ready"
    
    @patch('app.api.emotion_routes.classifier')
    def test_get_available_emotions_success(self, mock_classifier):
        """Test successful retrieval of available emotions."""
        # Setup mock
        mock_classifier.is_loaded = True
        mock_emotions = ["anger", "fear", "joy", "love", "sadness", "surprise"]
        mock_classifier.get_available_emotions.return_value = mock_emotions
        mock_classifier.model_name = "jitesh/emotion-english"
        
        # Test request
        response = client.get(f"{self.base_url}/emotions")
        
        # Assertions
        assert response.status_code == 200
        data = response.json()
        assert data["emotions"] == mock_emotions
        assert data["total_emotions"] == 6
        assert data["model_name"] == "jitesh/emotion-english"
    
    @patch('app.api.emotion_routes.classifier')
    def test_get_available_emotions_model_not_loaded(self, mock_classifier):
        """Test getting available emotions when model is not loaded."""
        # Setup mock
        mock_classifier.is_loaded = False
        
        # Test request
        response = client.get(f"{self.base_url}/emotions")
        
        # Assertions
        assert response.status_code == 503
        data = response.json()
        assert "Model is not loaded" in data["error"]
    
    @patch('app.api.emotion_routes.classifier')
    def test_health_check_healthy(self, mock_classifier):
        """Test health check when service is healthy."""
        # Setup mock
        mock_classifier.is_loaded = True
        
        # Test request
        response = client.get(f"{self.base_url}/health")
        
        # Assertions
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True
        assert "timestamp" in data
    
    @patch('app.api.emotion_routes.classifier')
    def test_health_check_loading(self, mock_classifier):
        """Test health check when model is loading."""
        # Setup mock
        mock_classifier.is_loaded = False
        
        # Test request
        response = client.get(f"{self.base_url}/health")
        
        # Assertions
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "loading"
        assert data["model_loaded"] is False
        assert "timestamp" in data
    
    @patch('app.api.emotion_routes.classifier')
    def test_reload_model_success(self, mock_classifier):
        """Test successful model reload."""
        # Setup mock
        mock_classifier.load_model.return_value = None
        
        # Test request
        response = client.post(f"{self.base_url}/model/reload")
        
        # Assertions
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Model reloaded successfully"
        assert data["status"] == "success"
        assert "timestamp" in data
    
    @patch('app.api.emotion_routes.classifier')
    def test_reload_model_failure(self, mock_classifier):
        """Test model reload failure."""
        # Setup mock
        mock_classifier.load_model.side_effect = Exception("Load failed")
        
        # Test request
        response = client.post(f"{self.base_url}/model/reload")
        
        # Assertions
        assert response.status_code == 500
        data = response.json()
        assert "Failed to reload model" in data["error"]


class TestGeneralRoutes:
    """Test cases for general API routes."""
    
    def test_root_redirect(self):
        """Test root endpoint redirects to docs."""
        response = client.get("/", allow_redirects=False)
        assert response.status_code == 307
        assert response.headers["location"] == "/docs"
    
    def test_api_info(self):
        """Test API info endpoint."""
        response = client.get("/info")
        
        # Assertions
        assert response.status_code == 200
        data = response.json()
        assert data["title"] == "Emotion Classification API"
        assert "version" in data
        assert data["docs_url"] == "/docs"
        assert data["redoc_url"] == "/redoc"
        assert data["openapi_url"] == "/openapi.json"


class TestErrorHandling:
    """Test cases for error handling."""
    
    @patch('app.api.emotion_routes.classifier')
    def test_internal_server_error(self, mock_classifier):
        """Test internal server error handling."""
        # Setup mock to raise unexpected exception
        mock_classifier.is_loaded = True
        mock_classifier.predict.side_effect = RuntimeError("Unexpected error")
        
        # Test request
        response = client.post(
            "/api/v1/classify",
            json={"text": "I am happy!"}
        )
        
        # Assertions
        assert response.status_code == 500
        data = response.json()
        assert "Internal server error" in data["error"]
        assert data["status_code"] == 500
    
    @patch('app.api.emotion_routes.classifier')
    def test_value_error_handling(self, mock_classifier):
        """Test ValueError handling in classification."""
        # Setup mock to raise ValueError
        mock_classifier.is_loaded = True
        mock_classifier.predict.side_effect = ValueError("Invalid input")
        
        # Test request
        response = client.post(
            "/api/v1/classify",
            json={"text": "I am happy!"}
        )
        
        # Assertions
        assert response.status_code == 400
        data = response.json()
        assert "Invalid input" in data["error"]


class TestResponseSchemas:
    """Test cases for response schema validation."""
    
    @patch('app.api.emotion_routes.classifier')
    def test_classification_response_schema(self, mock_classifier):
        """Test that classification response matches expected schema."""
        # Setup mock
        mock_classifier.is_loaded = True
        mock_predictions = [
            {'label': 'joy', 'score': 0.85},
            {'label': 'sadness', 'score': 0.15}
        ]
        mock_classifier.predict.return_value = mock_predictions
        
        # Test request
        response = client.post(
            "/api/v1/classify",
            json={"text": "I am happy!"}
        )
        
        # Assertions
        assert response.status_code == 200
        data = response.json()
        
        # Check required fields
        required_fields = ["text", "predictions", "top_emotion", "confidence"]
        for field in required_fields:
            assert field in data
        
        # Check prediction structure
        for prediction in data["predictions"]:
            assert "label" in prediction
            assert "score" in prediction
            assert isinstance(prediction["score"], float)
            assert 0 <= prediction["score"] <= 1
    
    @patch('app.api.emotion_routes.classifier')
    def test_batch_response_schema(self, mock_classifier):
        """Test that batch response matches expected schema."""
        # Setup mock
        mock_classifier.is_loaded = True
        mock_batch_predictions = [
            [{'label': 'joy', 'score': 0.8}],
            [{'label': 'sadness', 'score': 0.7}]
        ]
        mock_classifier.predict_batch.return_value = mock_batch_predictions
        
        # Test request
        response = client.post(
            "/api/v1/classify/batch",
            json={"texts": ["I am happy!", "I am sad."]}
        )
        
        # Assertions
        assert response.status_code == 200
        data = response.json()
        
        # Check required fields
        assert "results" in data
        assert "total_processed" in data
        assert isinstance(data["total_processed"], int)
        assert len(data["results"]) == data["total_processed"]


# Pytest fixtures
@pytest.fixture
def mock_loaded_classifier():
    """Fixture providing a mock loaded classifier."""
    classifier = Mock(spec=EmotionClassifier)
    classifier.is_loaded = True
    classifier.predict.return_value = [
        {'label': 'joy', 'score': 0.8},
        {'label': 'sadness', 'score': 0.2}
    ]
    classifier.predict_top_emotion.return_value = {'label': 'joy', 'score': 0.8}
    classifier.predict_batch.return_value = [
        [{'label': 'joy', 'score': 0.8}]
    ]
    classifier.get_model_info.return_value = {
        'model_name': 'jitesh/emotion-english',
        'is_loaded': 'True',
        'device': 'cpu'
    }
    return classifier


@pytest.fixture
def sample_texts():
    """Fixture providing sample texts for testing."""
    return [
        "I am very happy today!",
        "This is so sad and disappointing.",
        "I'm feeling excited about the future!",
        "This makes me really angry.",
        "I love spending time with my family.",
        "That was quite surprising!"
    ] 