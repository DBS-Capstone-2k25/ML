# Dockerfile

# --- Tahap 1: Base Image ---
# Kita mulai dari image resmi Python versi 3.11 yang ramping (slim).
# Ini adalah dasar yang bagus dan ukurannya tidak terlalu besar.
FROM python:3.11-slim

# --- Tahap 2: Konfigurasi Lingkungan ---
# Menetapkan direktori kerja di dalam container.
# Semua perintah selanjutnya akan dijalankan dari folder /app.
WORKDIR /app

# --- Tahap 3: Instalasi Dependensi ---
# Salin file requirements.txt terlebih dahulu.
# Docker akan menyimpan hasil layer ini di cache, jadi jika file ini tidak berubah,
# proses instalasi tidak akan diulang, membuat build selanjutnya lebih cepat.
COPY requirements.txt .

# Jalankan pip untuk menginstal semua library yang dibutuhkan dari requirements.txt.
RUN pip install --no-cache-dir -r requirements.txt

# --- Tahap 4: Salin Kode Aplikasi dan Model ---
# Salin semua file dari direktori lokal Anda (termasuk main.py dan folder adapter)
# ke dalam direktori /app di dalam container.
COPY . .

# --- Tahap 5: Konfigurasi Jaringan & Perintah Jalan ---
# Beritahu Docker bahwa container ini akan "mendengarkan" di port 8000.
EXPOSE 8000

# Perintah yang akan dijalankan saat container pertama kali dimulai.
# Ini akan menjalankan server Uvicorn untuk aplikasi FastAPI kita.
# --host 0.0.0.0 penting agar server bisa diakses dari luar container.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]