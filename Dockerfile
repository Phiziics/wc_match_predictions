FROM python:3.11-slim

WORKDIR /app

# Prevent Python from writing pycache files
ENV PYTHONDONTWRITEBYTECODE=1

# Prevent Python from buffering logs
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better Docker caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Streamlit port
EXPOSE 8502

# Run Streamlit
CMD ["streamlit", "run", "deployment/app_streamlit.py", "--server.address=0.0.0.0", "--server.port=8502"]