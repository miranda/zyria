#!/bin/bash

# Model information
MODEL_URL="https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q5_K_M.gguf"
MODEL_DIR="models/llama-3.2-3b"
MODEL_FILE="$MODEL_DIR/Llama-3.2-3B-Instruct-Q5_K_M.gguf"

# Check for curl
if ! command -v curl &> /dev/null; then
    echo "Error: curl is not installed. Please install curl to proceed."
    exit 1
fi

source venv/bin/activate

# Install Python dependencies
if [ -f "requirements.txt" ]; then
    echo "Installing Python dependencies from requirements.txt..."
    python3 -m pip install --upgrade pip
    python3 -m pip install -r requirements.txt
else
    echo "Warning: requirements.txt not found. Skipping Python dependency installation."
fi

echo "Downloading NLTK words corpus..."
python3 -c "import nltk; nltk.download('words')"

# Download the model
echo "Setting up model directory: $MODEL_DIR"
mkdir -p "$MODEL_DIR"

if [ ! -f "$MODEL_FILE" ]; then
    echo "Downloading model from $MODEL_URL..."
    curl -L -o "$MODEL_FILE" "$MODEL_URL"
    echo "Model downloaded to $MODEL_FILE."
else
    echo "Model already exists at $MODEL_FILE. Skipping download."
fi

echo "Setup complete!"
