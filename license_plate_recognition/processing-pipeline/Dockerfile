# Use the official Python image as the base image
FROM python:3.12

# Set the working directory in the container
WORKDIR /app

# Copy the project files into the container
COPY . /app

# Install the required packages
RUN pip install --no-cache-dir -r requirements.txt

# Install OpenCV dependencies
RUN apt-get update && apt-get install -y \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Update package lists and install FFmpeg
RUN apt-get -y update
RUN apt-get -y upgrade
RUN apt-get install -y ffmpeg

# Install Tesseract OCR
RUN apt-get update && apt-get install -y tesseract-ocr tesseract-ocr-eng && rm -rf /var/lib/apt/lists/*

# Set the environment variable for Tesseract data
ENV TESSDATA_PREFIX=/usr/share/tesseract-ocr/5/tessdata

# Expose the port for the video stream (if applicable)
EXPOSE 23000

# Set the entry point for the container
CMD ["python", "main.py"]