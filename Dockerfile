# Use an official Python runtime as a parent image
FROM python:3.9

# Install libgl1-mesa-glx to resolve libGL.so.1 dependency
RUN apt-get update && apt-get install -y libgl1-mesa-glx && apt-get install -y libglib2.0-0 && apt-get install -y ffmpeg


# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt

# Run script.py when the container launches
CMD ["python", "script.py"]
