# --- Tahap 1: Base Image ---
# Menggunakan image Python resmi versi 3.10 yang ringan (slim)
FROM python:3.10-slim

# --- Tahap 2: Konfigurasi Environment ---
# Menetapkan direktori kerja di dalam container
WORKDIR /code

# --- Tahap 3: Instalasi Dependensi ---
# Menyalin file requirements.txt terlebih dahulu untuk optimasi cache Docker
COPY requirements.txt .

# Menjalankan instalasi pustaka dari requirements.txt
# --no-cache-dir digunakan untuk menjaga ukuran image tetap kecil
RUN  pip install --default-timeout=100 --no-cache-dir -r requirements.txt

# --- Tahap 4: Menyalin Kode Aplikasi ---
# Menyalin direktori 'app' yang berisi logika utama
COPY ./app /code/app
# Menyalin direktori 'model' yang berisi file best.onnx
COPY ./model /code/model

# --- Tahap 5: Menjalankan Aplikasi ---
# Memberi tahu Docker bahwa container akan berjalan di port 8000
EXPOSE 8000

# Perintah default untuk menjalankan aplikasi saat container dimulai
# --host 0.0.0.0 membuat server dapat diakses dari luar container
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
