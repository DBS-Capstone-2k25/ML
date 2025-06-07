# --- Tahap 1: Base Image ---
# Menggunakan image Python resmi versi 3.10 yang ringan (slim)
FROM python:3.10-slim

# --- Tahap 2: Konfigurasi Environment ---
# Menetapkan direktori kerja di dalam container
WORKDIR /code

# --- Tahap 3: Instalasi Dependensi ---

# !! BAGIAN BARU !!
# Instal dependensi level sistem operasi yang dibutuhkan oleh OpenCV dan dependensi lainnya
# Ini untuk memperbaiki error "ImportError: libGL.so.1" dan "libgthread-2.0.so.0"
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Upgrade pip terlebih dahulu
RUN pip install --upgrade pip

# Menyalin file requirements.txt
COPY requirements.txt .

# Menjalankan instalasi pustaka dari requirements.txt
RUN pip install \
    --default-timeout=60 \
    --no-cache-dir \
    -r requirements.txt

# --- Tahap 4: Membuat Pengguna Non-Root ---
RUN useradd -ms /bin/bash appuser

# --- Tahap 5: Menyalin Kode Aplikasi ---
COPY ./app /code/app
COPY ./model /code/model
RUN chown -R appuser:appuser /code

# --- Tahap 6: Menjalankan Aplikasi sebagai Pengguna Non-Root ---
USER appuser
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
