# ===================================================================
#  TAHAP 1: BUILDER - Tahap untuk menginstall semua dependensi
# ===================================================================
# Menggunakan base image Python resmi yang ringan (bukan CUDA)
FROM python:3.10-slim as builder

# Menetapkan direktori kerja di dalam container
WORKDIR /app

# Menyalin file requirements terlebih dahulu untuk memanfaatkan caching Docker.
# Build tidak akan menginstall ulang dependensi jika file ini tidak berubah.
COPY requirements.txt .

# Menginstall dependensi dari requirements.txt
# --no-cache-dir membuat image sedikit lebih kecil
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ===================================================================
#  TAHAP 2: FINAL - Tahap untuk menjalankan aplikasi
# ===================================================================
# Memulai lagi dari base image yang bersih dan kecil untuk produksi
FROM python:3.10-slim

# Menetapkan direktori kerja
WORKDIR /app

# Membuat user non-root bernama "appuser" untuk keamanan.
# Menjalankan container sebagai root adalah praktik yang buruk.
RUN useradd --system --create-home appuser
USER appuser

# Menyalin dependensi yang sudah ter-install dari tahap 'builder'
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Menyalin semua file dari direktori Anda ke dalam direktori /app di container.
# Ini termasuk main.py, requirements.txt, dan folder model 'qwen_chatbot_...'.
COPY . .

# Memberitahu Cloud Run bahwa container akan menerima traffic di port 8080
EXPOSE 8080

# Perintah untuk menjalankan server aplikasi saat container dimulai.
# Perintah ini akan menjalankan file main.py Anda menggunakan uvicorn.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
