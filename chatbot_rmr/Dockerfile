# Gunakan base image resmi Python 3.10 yang ramping (slim).
# Image ini jauh lebih kecil karena tidak menyertakan driver CUDA.
FROM python:3.10-slim

# Atur agar tidak ada prompt interaktif selama build
ENV DEBIAN_FRONTEND=noninteractive

# Instal dependensi sistem yang mungkin diperlukan (opsional, tapi aman untuk disertakan)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Atur direktori kerja di dalam container
WORKDIR /app

# Salin file requirements terlebih dahulu untuk memanfaatkan Docker layer caching.
COPY requirements.txt .

# Instal Pytorch versi CPU.
RUN python3 -m pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Instal sisa dependensi Python.
RUN python3 -m pip install --no-cache-dir -r requirements.txt

# Salin seluruh kode aplikasi, termasuk direktori model, ke dalam container.
# File yang ada di .dockerignore akan dikecualikan.
COPY . .

# Beri tahu Docker bahwa container akan listen di port 8000.
EXPOSE 8000

# Perintah untuk menjalankan aplikasi saat container dimulai.
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]