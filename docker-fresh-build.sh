#!/bin/bash
# Fresh Docker build script without cache

echo "🧹 Cleaning Docker cache and unused objects..."
docker system prune -a -f

echo "🐳 Building Docker image without cache..."
docker build --no-cache --rm -t emotion-classification-api .

echo "✅ Fresh build complete!"
echo "🚀 To run the container:"
echo "   docker run -p 8000:8000 emotion-classification-api" 