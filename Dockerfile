FROM python:3.11-slim
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# Create directories if they don't exist
RUN mkdir -p checkpoints vector_store data
EXPOSE 7860
# Start the Flask server on HF's required port
CMD ["python", "04_chatbot.py", "--api", "--port", "7860"]