# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Install system dependencies for building Python packages, psycopg2, and Playwright
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    curl \
    git \
    wget \
    # Playwright dependencies
    libglib2.0-0 \
    libnss3 \
    libnspr4 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libcups2 \
    libdrm2 \
    libdbus-1-3 \
    libxcb1 \
    libxkbcommon0 \
    libx11-6 \
    libxcomposite1 \
    libxdamage1 \
    libxext6 \
    libxfixes3 \
    libxrandr2 \
    libgbm1 \
    libpango-1.0-0 \
    libcairo2 \
    libasound2 \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /workspace

# Note: Python package installation moved to devcontainer.json postCreateCommand
# This allows for better development workflow and dependency management

# Expose the Streamlit default port (8501)
EXPOSE 8501

# Keep the container running (you can override this later via docker-compose)
CMD [ "sleep", "infinity" ]
