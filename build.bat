@echo off
echo Building Emotion Classification API Docker image...
docker build -t emotion-classification-api .

echo.
echo Build complete! 
echo.
echo To run the container:
echo docker run -p 8000:8000 emotion-classification-api
echo.
echo Or use docker-compose:
echo docker-compose up -d
echo.
pause 