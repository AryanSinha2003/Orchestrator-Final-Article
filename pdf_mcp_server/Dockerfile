FROM python:3.11-slim

# Install system dependencies (needed for unstructured/pdf parsing and OpenCV)
RUN apt-get update && apt-get install -y \
    poppler-utils \
    libmagic-dev \
    tesseract-ocr \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create cache directory for FlashRank to avoid permission issues
RUN mkdir -p /tmp/flashrank_cache && chmod 777 /tmp/flashrank_cache

# Copy the rest of the application
COPY . .

# Expose Hugging Face's default port
EXPOSE 7860

# Force HF to use 7860 as the PORT environment variable
ENV PORT=7860

# Run the server
CMD ["python", "mcp_server.py"]
