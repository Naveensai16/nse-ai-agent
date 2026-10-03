FROM python:3.11-slim

# Configure environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    PORT=7860 \
    OLLAMA_BASE_URL=http://127.0.0.1:11434 \
    OLLAMA_MODEL=llama3.2:1b \
    OLLAMA_TIMEOUT_SECONDS=25.0 \
    OLLAMA_MAX_PROMPT_LENGTH=1500 \
    OLLAMA_MAX_CONCURRENCY=2 \
    DATA_DIR=/home/user/app/data \
    HOME=/home/user

# Install system dependencies, curl, and procps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    procps \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install official Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Hugging Face Spaces requires a non-root user with UID 1000
RUN useradd -m -u 1000 user

WORKDIR /home/user/app

# Pre-create required writable directories
RUN mkdir -p /home/user/app/data /home/user/.ollama && \
    chown -R user:user /home/user

# Switch to non-root user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

# Install Python requirements
COPY --chown=user:user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Copy all application files
COPY --chown=user:user . .

# Ensure entrypoint script is executable
RUN chmod +x /home/user/app/entrypoint.sh

# Pre-download the LLM model during build phase for instant container startup
RUN ollama serve & \
    until curl -s http://127.0.0.1:11434 > /dev/null; do sleep 1; done && \
    ollama pull llama3.2:1b && \
    pkill -f "ollama serve"

# Expose default Hugging Face Spaces web port
EXPOSE 7860

ENTRYPOINT ["/home/user/app/entrypoint.sh"]
