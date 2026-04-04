# NeuralILT-DSCNN Docker image
# Provides a reproducible environment for training and evaluation
#
# Build:  docker build -t neuralilt-dscnn .
# Run:    docker run -it -v $(pwd)/data:/app/data -v $(pwd)/results:/app/results neuralilt-dscnn

FROM python:3.11-slim

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Default command
CMD ["bash"]
