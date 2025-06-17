# Use official Python 3.11 image
FROM python:3.11.8-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set work directory
WORKDIR /app

# Install system dependencies (e.g., ffmpeg)
RUN apt-get update && \
    apt-get install -y ffmpeg gcc && \
    apt-get clean

# Install Python dependencies
COPY requirements.txt /app/
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy project files
COPY . /app/

# Run Streamlit app (adjust if using Flask or other)
CMD ["streamlit", "run", "app.py", "--server.port=10000", "--server.enableCORS=false"]
