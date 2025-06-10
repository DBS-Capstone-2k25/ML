# Gunakan base image resmi dari NVIDIA dengan CUDA 12.1, cocok untuk PyTorch
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Hindari prompt interaktif saat instalasi
ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.10 dan pip
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Set direktori kerja di dalam container
WORKDIR /app

# Salin file requirements terlebih dahulu untuk caching
COPY requirements.txt .

# Install semua dependensi Python
RUN pip3 install --no-cache-dir -r requirements.txt

# Salin semua file proyek (kode dan model) ke dalam container
COPY . .

# Beritahu Docker port mana yang akan digunakan oleh aplikasi
EXPOSE 8080

# Perintah untuk menjalankan server FastAPI saat container dimulai
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]