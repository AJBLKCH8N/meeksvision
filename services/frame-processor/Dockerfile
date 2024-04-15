# Start from a base Python image
FROM python:3.9-slim

# Set environment variables to reduce Python bytecode generation and unbuffered outputs
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
# This includes libraries needed by dlib, OpenCV, and face_recognition
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       libopencv-dev \
       cmake \
       g++ \
       libsm6 \
       libxext6 \
       libxrender-dev \
       libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# It's important to make sure that dlib is properly installed,
# you might need to build dlib from source if not using pip.
# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application files into the container
COPY app/ .

# Command to run the application
CMD ["python", "main.py"]
