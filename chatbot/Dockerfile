# Gunakan base image resmi Python
FROM python:3.10-slim

# Set direktori kerja di dalam container
WORKDIR /app

# Salin file requirements terlebih dahulu untuk caching
COPY requirements.txt .

# Install semua dependensi Python
RUN pip install --no-cache-dir -r requirements.txt

# Salin semua file proyek (kode dan model) ke dalam container
COPY . .

# Beritahu Docker port mana yang akan digunakan oleh aplikasi
EXPOSE 8080

# Perintah untuk menjalankan server FastAPI saat container dimulai
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
