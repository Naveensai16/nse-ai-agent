#!/bin/bash
set -e

echo "=== [1/3] Starting internal Ollama service ==="
ollama serve &

# Wait for Ollama service to respond on 127.0.0.1:11434
echo "=== Waiting for Ollama to become available ==="
until curl -s http://127.0.0.1:11434 > /dev/null; do
    sleep 1
done
echo "=== Ollama service is active ==="

# Check if model exists, pull if missing
echo "=== [2/3] Checking LLM model: ${OLLAMA_MODEL:-llama3.2:1b} ==="
if ! ollama list | grep -q "${OLLAMA_MODEL:-llama3.2:1b}"; then
    echo "Model ${OLLAMA_MODEL:-llama3.2:1b} not found locally, pulling..."
    ollama pull "${OLLAMA_MODEL:-llama3.2:1b}"
else
    echo "Model ${OLLAMA_MODEL:-llama3.2:1b} is ready."
fi

# Ensure data directory exists
mkdir -p /home/user/app/data

echo "=== [3/3] Starting NSE AI Agent Streamlit on port ${PORT:-7860} ==="
exec streamlit run app.py \
    --server.port="${PORT:-7860}" \
    --server.address="0.0.0.0" \
    --server.headless=true \
    --browser.gatherUsageStats=false
