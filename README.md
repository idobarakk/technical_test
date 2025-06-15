# Emotion Classification API

A powerful and efficient FastAPI-based web service for classifying emotions in text using state-of-the-art HuggingFace transformer models.

## Features

- **Single Text Classification**: Classify emotions in individual texts
- **Top Emotion Only**: Get just the most likely emotion for efficiency
- **Model Information**: Access details about the loaded model
- **Health Monitoring**: Check service health and model status
- **Comprehensive Error Handling**: Detailed error responses with proper HTTP status codes
- **Interactive API Documentation**: Automatic Swagger UI and ReDoc generation
- **Comprehensive Testing**: Full test coverage with pytest

## Supported Emotions

The model can classify the following emotions:
- Anger
- Cheeky
- Confuse  
- Curious
- Disgust
- Empathetic
- Energetic
- Fear
- Grumpy
- Guilty
- Impatient
- Joy
- Love
- Neutral
- Sadness
- Serious
- Surprise
- Suspicious
- Think
- Whiny

*Total: 20 different emotions*

## Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd emotion-classification-api
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the server:
```bash
python main.py
```

The API will be available at `http://localhost:8000`

### Alternative: Using uvicorn directly

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

## Docker Deployment

### Prerequisites

- Docker Desktop installed on Windows
- Docker Compose (included with Docker Desktop)

### Quick Docker Setup

1. **Clone the repository:**
```bash
git clone <repository-url>
cd emotion-classification-api
```

2. **Build and run with Docker Compose (Recommended):**
```bash
docker-compose up -d
```

3. **Or build and run manually:**
```bash
# Build the image
docker build -t emotion-classification-api .

# Run the container
docker run -p 8000:8000 emotion-classification-api
```

### Windows Batch Scripts

For Windows users, use the provided batch files:

- **`build.bat`** - Build the Docker image
- **`run.bat`** - Start the container with docker-compose

Simply double-click the batch files or run them from Command Prompt.

### Docker Commands

```bash
# Start the service
docker-compose up -d

# Stop the service
docker-compose down

# View logs
docker-compose logs -f

# Rebuild after changes
docker-compose up -d --build

# Check container status
docker-compose ps
```

### Docker Configuration

The Docker setup includes:
- **Multi-stage build** for optimized image size
- **Non-root user** for security
- **Health checks** for monitoring
- **Volume mounting** for persistent logs
- **Environment variables** for configuration

**Default URLs:**
- API: http://localhost:8000
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## API Documentation

Once the server is running, you can access:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI JSON**: http://localhost:8000/openapi.json

## API Endpoints

### Text Classification

#### Single Text Classification
```http
POST /api/v1/classify
Content-Type: application/json

{
  "text": "I am very happy today!"
}
```

**Response:**
```json
{
  "text": "I am very happy today!",
  "predictions": [
    {"label": "joy", "score": 0.85},
    {"label": "sadness", "score": 0.15}
  ],
  "top_emotion": "joy",
  "confidence": 0.85
}
```

#### Top Emotion Only
```http
POST /api/v1/classify/top
Content-Type: application/json

{
  "text": "I am very happy today!"
}
```

**Response:**
```json
{
  "text": "I am very happy today!",
  "emotion": "joy",
  "confidence": 0.85
}
```

### Utility Endpoints

#### Health Check
```http
GET /api/v1/health
```

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "timestamp": "2024-01-01T12:00:00.000Z"
}
```

#### Model Information
```http
GET /api/v1/model/info
```

**Response:**
```json
{
  "model_name": "jitesh/emotion-english",
  "is_loaded": "True",
  "device": "cpu",
  "status": "ready"
}
```

#### Available Emotions
```http
GET /api/v1/emotions
```

**Response:**
```json
{
  "emotions": [
    "anger", "cheeky", "confuse", "curious", "disgust", 
    "empathetic", "energetic", "fear", "grumpy", "guilty",
    "impatient", "joy", "love", "neutral", "sadness", 
    "serious", "surprise", "suspicious", "think", "whiny"
  ],
  "total_emotions": 20,
  "model_name": "jitesh/emotion-english"
}
```

## Usage Examples

### Python Client Example

```python
import requests

# Single text classification
response = requests.post(
    "http://localhost:8000/api/v1/classify",
    json={"text": "I can't wait any longer!"}
)
result = response.json()
print(f"Top emotion: {result['top_emotion']} (confidence: {result['confidence']:.2f})")

# Get available emotions
response = requests.get(
    "http://localhost:8000/api/v1/emotions"
)
emotions = response.json()
print(f"Available emotions ({emotions['total_emotions']}): {', '.join(emotions['emotions'])}")

# Model info
response = requests.get(
    "http://localhost:8000/api/v1/model/info"
)
model_info = response.json()
print(f"Model: {model_info['model_name']} (Status: {model_info['status']})")
```

### cURL Examples

```bash
# Single classification
curl -X POST "http://localhost:8000/api/v1/classify" \
     -H "Content-Type: application/json" \
     -d '{"text": "I am feeling great today!"}'

# Health check
curl -X GET "http://localhost:8000/api/v1/health"

# Model info
curl -X GET "http://localhost:8000/api/v1/model/info"

# Available emotions
curl -X GET "http://localhost:8000/api/v1/emotions"
```

## Testing

The project includes comprehensive tests covering unit tests, integration tests, and API endpoint tests.

### Running Tests

```bash
# Run all tests
pytest

# Run only unit tests (fast)
pytest -m "not integration"

# Run with coverage
pytest --cov=app --cov-report=html

# Run specific test file
pytest tests/test_emotion_classifier.py

# Run with verbose output
pytest -v
```

### Test Categories

- **Unit Tests**: Test individual components in isolation
- **Integration Tests**: Test complete workflows (require model download)
- **API Tests**: Test HTTP endpoints and responses

## Project Structure

```
emotion-classification-api/
├── app/
│   ├── __init__.py
│   ├── api/
│   │   ├── __init__.py
│   │   └── emotion_routes.py      # API route definitions
│   ├── models/
│   │   ├── __init__.py
│   │   └── emotion_classifier.py  # Core classification logic
│   └── schemas/
│       ├── __init__.py
│       └── emotion_schemas.py     # Pydantic schemas
├── tests/
│   ├── __init__.py
│   ├── test_emotion_classifier.py # Unit tests
│   └── test_api_routes.py         # API integration tests
├── main.py                        # FastAPI application
├── requirements.txt               # Python dependencies
├── pytest.ini                    # Pytest configuration
└── README.md                      # This file
```

## Configuration

### Environment Variables

You can configure the application using environment variables:

- `LOG_LEVEL`: Set logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- `MODEL_NAME`: Override the default model name
- `HOST`: Server host (default: 0.0.0.0)
- `PORT`: Server port (default: 8000)

### Production Deployment

For production deployment, consider:

1. **Security**: Configure CORS origins properly
2. **Performance**: Use multiple workers with Gunicorn
3. **Monitoring**: Set up proper logging and monitoring
4. **Caching**: Implement model caching strategies

Example production command:
```bash
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

## Model Information

- **Model**: jitesh/emotion-english
- **Framework**: HuggingFace Transformers
- **Type**: BERT-based sequence classification
- **Size**: ~400MB (downloaded on first run)

## Development

### Code Quality

The project follows Python best practices:
- Type hints throughout
- Comprehensive docstrings
- Proper logging instead of print statements
- Object-oriented design
- Comprehensive error handling
- Full test coverage

### Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Run the test suite
6. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support or questions, please open an issue in the repository or contact the development team.

## Changelog

### v1.0.0
- Initial release
- Complete API implementation
- Comprehensive test suite
- Full documentation