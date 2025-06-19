# Ultra-lightweight Docker image for VADER emotion classification API
FROM python:3.10-slim

# Install minimal system dependencies (only curl for health check)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Environment variables for Python optimization
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .

# Install Python dependencies (ultra-lightweight, only VADER + FastAPI)
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Test the VADER sentiment analyzer
RUN python -c "from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer; analyzer = SentimentIntensityAnalyzer(); print('VADER installed successfully'); print('Test:', analyzer.polarity_scores('I am happy'))"

# Copy application files
COPY main.py .

# Create non-root user for security
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Health check with very fast intervals since startup is instant
HEALTHCHECK --interval=10s --timeout=2s --start-period=2s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start the application
CMD ["python", "main.py"] 