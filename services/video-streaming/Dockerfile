# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory to /app in the container
WORKDIR /app

# Install any necessary dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY app/ .

# Command to run the application
CMD ["python", "main.py"]
