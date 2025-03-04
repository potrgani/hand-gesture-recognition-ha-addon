# Use an official Python runtime as a parent image
FROM python:3.9

# Prevents Python from buffering stdout/stderr (for real-time logs)
ENV PYTHONUNBUFFERED=1

# Install necessary dependencies (OpenGL, FFmpeg, and OpenCV)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Run script.py when the container launches
ENTRYPOINT ["python", "script.py"]
