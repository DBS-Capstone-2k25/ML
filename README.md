# Machine Learning Models for Bedah Sampah 

Repositori ini berisi kode sumber untuk proyek Bedah Sampah, yang terdiri dari tiga layanan berbasis Machine Learning yaitu image classification, object-detection, dan juga chatbot. Setiap layanan dirancang untuk melakukan tugas spesifik terkait identifikasi dan informasi sampah yang terdokumentasi secara terpisah dalam setiap endpoint yang berbentuk Swagger API dengan dibangun menggunakan FastAPI.

## 🗂️ Struktur Repositori

Proyek ini menggunakan struktur yang setiap layanan berada di dalam direktorinya masing-masing dengan struktur spesifik sebagai berikut:

-   **/chatbot**:
    -   `main.py`: Titik masuk utama untuk server API.
    -   `inference.py`: Logika untuk menjalankan inferensi model.
    -   `qwen_1.5B_chatbot_p100_final/`: Direktori yang berisi file model chatbot.
    -   `Dockerfile` & `requirements.txt`: Konfigurasi deployment.

-   **/image-classification**:
    -   `/deploy_model`: **Direktori ini berisi terkait FastAPI.**
        -   `/app`: Modul aplikasi FastAPI.
        -   `/model`: Direktori untuk menyimpan file model klasifikasi.
        -   `Dockerfile` & `requirements.txt`: Konfigurasi deployment.

-   **/object-detection**:
    -   `/app`: Modul aplikasi FastAPI, berisi `main.py`.
    -   `/model`: Direktori untuk menyimpan file model deteksi (YOLO).
    -   `/hasil_deteksi`: Direktori untuk menyimpan output gambar.
    -   `Dockerfile` & `requirements.txt`: Konfigurasi deployment.

## ✨ Fitur & Layanan

### 1. API Klasifikasi Sampah
Layanan ini menerima sebuah gambar dan mengklasifikasikan jenis sampah yang ada di dalamnya (misalnya: organik, plastik, kertas, dll.).

-   **Platform Host**: Google Cloud Run
-   **Endpoint**: `POST /predict`
-   **Request Body**: `multipart/form-data`
    -   **`file`**: File gambar yang akan diprediksi (wajib).
-   **Contoh Penggunaan (`curl`)**:
    ```bash
    curl -X POST "[https://ml-models-106990625306.asia-southeast2.run.app/predict](https://ml-models-106990625306.asia-southeast2.run.app/predict)" \
         -H "accept: application/json" \
         -H "Content-Type: multipart/form-data" \
         -F "file=@/path/to/your/image.jpg"
    ```
-   **Respon Sukses (200 OK)**: Mengembalikan hasil prediksi dalam format JSON.
    ```json
    {
      "prediction": "Plastik",
      "confidence_score": 0.95
    }
    ```
    *(Catatan: Struktur respon dapat bervariasi)*

### 2. API Deteksi Objek YOLO
Layanan ini menggunakan model YOLO untuk mendeteksi lokasi objek-objek sampah dalam sebuah gambar dan memberikan *bounding box* di sekelilingnya.

-   **Platform Host**: Google Cloud Run
-   **Endpoint**: `POST /detect`
-   **Request Body**: `multipart/form-data`
    -   **`file`**: File gambar yang akan dideteksi (wajib).
-   **Contoh Penggunaan (`curl`)**:
    ```bash
    curl -X POST "[https://ml-object-detection-106990625306.asia-southeast2.run.app/detect](https://ml-object-detection-106990625306.asia-southeast2.run.app/detect)" \
         -H "accept: application/json" \
         -H "Content-Type: multipart/form-data" \
         -F "file=@/path/to/your/image.png"
    ```
-   **Respon Sukses (200 OK)**: Mengembalikan gambar asli dengan bounding box yang sudah digambar di atasnya atau data JSON yang berisi koordinat objek.

### 3. API Chatbot Sampah
Layanan ini menyediakan antarmuka percakapan (chatbot) untuk menjawab pertanyaan pengguna terkait pengelolaan sampah, daur ulang, dan informasi relevan lainnya.

-   **Platform Host**: Google Cloud Run
-   **Endpoint**: `POST /chat`
-   **Request Body**: `application/json`
-   **Contoh Penggunaan (`curl`)**:
    ```bash
    curl -X POST "[https://chatbot-qwen-1069990625306.asia-southeast2.run.app/docs](https://chatbot-qwen-1069990625306.asia-southeast2.run.app/docs)" \
         -H "accept: application/json" \
         -H "Content-Type: application/json" \
         -d '{"prompt": "Bagaimana cara mendaur ulang botol plastik?", "max_new_tokens": 100}'
    ```
-   **Respon Sukses (200 OK)**: Mengembalikan jawaban dari chatbot dalam format JSON.
    ```json
    {
      "response": "Untuk mendaur ulang botol plastik, pertama pisahkan tutupnya, bersihkan botol dari sisa cairan, lalu kumpulkan di tempat sampah khusus plastik."
    }
    ```
    *(Catatan: Struktur respon dapat bervariasi)*

## 🛠️ Teknologi yang Digunakan

-   **Bahasa Pemrograman**: Python
-   **Framework API**: FastAPI
-   **Model**:
    -   Model Klasifikasi Gambar (Pretrained-Model MobileNetV2)
    -   Model Deteksi Objek (YOLOv12)
    -   Model Bahasa (Qwen2-1.5B-Instruct)
-   **Deployment**:
    -   Google Cloud Run
    -   Docker

## 🚀 Replikasi Proyek (Setup Lokal)

Ikuti langkah-langkah berikut untuk menjalankan setiap layanan di lingkungan lokal Anda.

### Prasyarat
-   [Git](https://git-scm.com/)
-   [Python](https://www.python.org/downloads/) 3.8+
-   [Docker](https://www.docker.com/products/docker-desktop/) (Opsional, untuk replikasi deployment)

### Langkah-langkah Setup

1.  **Clone Repositori**
    ```bash
    git clone [https://github.com/DBS-Capstone-2k25/ML.git](https://github.com/DBS-Capstone-2k25/ML.git)
    cd ML
    ```

2.  **Menjalankan Salah Satu Layanan (Contoh: Image Classification)**

    Setiap layanan harus di-setup secara terpisah.

    a. **Pindah ke Direktori Layanan**
    ```bash
    cd image-classification
    ```
    b. **Pindah ke direktori untuk menjalankan FastAPI**
    ```bash
    cd deploy_model
    ```
    c. **Buat dan Aktifkan Lingkungan Virtual (Virtual Environment)**
    ```bash
    # Membuat venv
    python -m venv venv

    # Mengaktifkan di MacOS/Linux
    source venv/bin/activate

    # Mengaktifkan di Windows
    .\venv\Scripts\activate
    ```

    d. **Install Dependensi**
    
    ```bash
    pip install -r requirements.txt
    ```

    d. **Jalankan Server API Lokal**
    API akan berjalan menggunakan `uvicorn`. Pastikan file utama Anda bernama `main.py` dan instance FastAPI bernama `app`.
    ```bash
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
    ```
    e. **Akses Dokumentasi API Lokal**
    Buka browser Anda dan kunjungi `http://127.0.0.1:8000/docs` untuk melihat antarmuka Swagger UI seperti pada screenshot.

4.  **Ulangi langkah 2** untuk direktori `/object-detection` dan `/chatbot` untuk menjalankan layanan lainnya (pastikan untuk menggunakan port dengan cara mengedit file.py-nya yang berbeda jika dijalankan bersamaan, misal: `--port 8001`, `--port 8002`).

### Deployment
-   **Google Cloud Run**: Layanan di-containerize menggunakan Docker, di-push ke Google Artifact Registry, lalu di-deploy sebagai layanan Cloud Run.
