@echo off
echo Starting Emotion Classification API...
docker-compose up -d

echo.
echo Container started!
echo API will be available at: http://localhost:8000
echo Swagger UI: http://localhost:8000/docs
echo.
echo To stop the container: docker-compose down
echo To view logs: docker-compose logs -f
echo.
pause 