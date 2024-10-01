# Use the official Python 3.12 slim image
FROM nvidia/cuda:12.6.1-cudnn-devel-ubuntu24.04

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-poetry \
    build-essential \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy the pyproject.toml and poetry.lock files
COPY pyproject.toml poetry.lock /app/

# Install project dependencies
RUN poetry install --no-root --no-interaction --no-ansi

# If flash-attn installation fails, try installing it separately
RUN poetry run pip install --no-cache-dir --use-pep517 --no-build-isolation "flash-attn (==2.6.3)" 

# Copy the application code
COPY . /app

# Expose the port that your app will run on
EXPOSE 44777

# Command to run the app
CMD ["poetry", "run", "uvicorn", "embeddrix.app:app", "--host", "0.0.0.0", "--port", "44777", "--lifespan", "on"]