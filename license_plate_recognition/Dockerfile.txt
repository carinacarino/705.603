# Use the official Python image as the base image
FROM python:3.12

# Set the working directory in the container
WORKDIR /app

# Copy the project files and the video file into the container
COPY . /app
COPY LicensePlateReaderSample_4K.mov /app/

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
RUN apt-get update && apt-get install -y tesseract-ocr && rm -rf /var/lib/apt/lists/*

# Set the environment variable for Tesseract data
ENV TESSDATA_PREFIX=/usr/share/tesseract-ocr/4.00/tessdata

# Expose the port for the video stream (if applicable)
EXPOSE 23000

# Set the entry point for the container
CMD ["sh", "-c", "ffmpeg -i LicensePlateReaderSample_4K.mov -vcodec mpeg4 -f mpegts udp://127.0.0.1:23000 & python main.py"]