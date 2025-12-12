# Use the official Python image from the Docker Hub
FROM tensorflow/tensorflow:latest-gpu

# Set the working directory in the container
WORKDIR /app

# Install Python packages
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt


# Copy the specific directories and files into the container
COPY src /app/src
COPY scripts /app/scripts

# Start an interactive terminal
CMD ["bash"]
