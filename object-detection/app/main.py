# app/main.py

from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware # Impor Middleware CORS
from fastapi.staticfiles import StaticFiles # Impor untuk menyajikan file statis
from PIL import Image
import io
import os
from ultralytics import YOLO
from datetime import datetime

# --- Inisialisasi Aplikasi ---
app = FastAPI(title="API Deteksi Objek YOLO")


# --- KONFIGURASI CORS (MENGIZINKAN SEMUA ORIGIN) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Mengizinkan semua domain
    allow_credentials=True,
    allow_methods=["*"], # Mengizinkan semua metode (GET, POST, dll.)
    allow_headers=["*"], # Mengizinkan semua header
)
# --- AKHIR DARI KONFIGURASI CORS ---


# --- Memuat Model ---
MODEL_PATH = os.path.join("model", "best.onnx")
try:
    model = YOLO(MODEL_PATH, task='detect')
    print(f"Model berhasil dimuat dari: {MODEL_PATH}")
except Exception as e:
    print(f"Gagal memuat model: {e}")
    model = None


# --- Menyajikan Folder 'hasil_deteksi' sebagai Folder Statis ---
OUTPUT_IMAGE_DIR = "hasil_deteksi"
os.makedirs(OUTPUT_IMAGE_DIR, exist_ok=True)
app.mount("/hasil-deteksi", StaticFiles(directory=OUTPUT_IMAGE_DIR), name="hasil_deteksi")


# --- API Endpoints ---

@app.get("/")
def read_root():
    return {"status": "ok", "pesan": "API YOLO berjalan dengan baik."}

@app.post("/detect/")
async def detect_objects(request: Request, file: UploadFile = File(...)):
    if not model:
        raise HTTPException(status_code=500, detail="Model tidak dapat dimuat. Periksa log server.")

    contents = await file.read()
    
    try:
        img = Image.open(io.BytesIO(contents))
        
        results = model(img)

        detections = []
        path_gambar_hasil = None

        if results:
            result = results[0]
            if result.boxes:
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    detections.append({
                        "nama_kelas": result.names[class_id],
                        "kepercayaan": float(box.conf[0]),
                        "kotak_pembatas": [int(coord) for coord in box.xyxy[0]]
                    })
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_filename = f"{timestamp}_{file.filename}"
                
                local_path_gambar_hasil = os.path.join(OUTPUT_IMAGE_DIR, output_filename)
                annotated_image = result.plot()
                annotated_image_rgb = Image.fromarray(annotated_image[...,::-1])
                annotated_image_rgb.save(local_path_gambar_hasil)

                path_gambar_hasil = str(request.url_for('hasil_deteksi', path=output_filename))

        return JSONResponse(content={
            "deteksi": detections,
            "gambar_hasil_url": path_gambar_hasil
        })

    except Exception as e:
        print(f"Terjadi error saat memproses gambar: {e}")
        raise HTTPException(status_code=500, detail=f"Gagal memproses gambar. Error: {str(e)}")

@app.get("/status")
def get_status():
    return {"model_loaded": model is not None}
