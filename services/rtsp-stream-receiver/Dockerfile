# Start with a lightweight Python base image
FROM python:3.9-slim

# Define environment variables for better security and performance
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set the working directory within the container
WORKDIR /app

# Install system dependencies
# This step ensures that the necessary libraries and tools are installed
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       libopencv-dev \
       build-essential \
       retry \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
# Copy the requirements file first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application's main directory into the container
COPY app/ .

# The command to run the application
CMD ["python", "main.py"]
