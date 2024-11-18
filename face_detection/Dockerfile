# Use Python 3.12 slim image as base
FROM python:3.12-slim

# Set environment variables to reduce warnings and issues
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    OPENCV_LOG_LEVEL=DEBUG

# Install system dependencies required for OpenCV with explicit versions
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libx11-6 \
    libgstreamer1.0-0 \
    libgstreamer-plugins-base1.0-0 \
    libgtk2.0-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies with specific versions and verbose output
RUN pip install --no-cache-dir --verbose \
    opencv-python-headless==4.9.0.80 \
    streamlit==1.31.0 \
    numpy==1.26.0

# Test OpenCV installation
RUN python -c "import cv2; print('OpenCV version:', cv2.__version__)"

# Copy the application code
COPY . .

# Set environment variable for Streamlit
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Expose the port
EXPOSE 8501

# Command to run the Streamlit app
CMD ["streamlit", "run", "--server.address", "0.0.0.0", "--server.port", "8501", "Streamlit.py"]
