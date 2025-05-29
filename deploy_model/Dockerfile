# Gunakan image dasar yang lebih ringan dan cocok untuk TensorFlow
FROM python:3.9-slim

# Install dependencies dasar
RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
 && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Salin semua file project
COPY ./app ./app
COPY ./model ./model
COPY requirements.txt .

# Install dependensi Python
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Expose port 8000 untuk FastAPI
EXPOSE 8000

# Jalankan FastAPI pakai uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
