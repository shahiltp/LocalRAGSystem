# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Install uv for faster package installation
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set the working directory in the container
WORKDIR /app

# Install system dependencies required for building packages
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Configure uv to use system Python (required for Docker)
ENV UV_SYSTEM_PYTHON=1

# Copy the requirements file into the container
COPY requirements-docker.txt ./requirements.txt

# Install any needed packages specified in requirements.txt using uv (much faster than pip)
RUN uv pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application's code into the container
COPY . .

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Run api.py when the container launches
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
