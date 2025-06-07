# app/main.py

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import io
import os
from ultralytics import YOLO
from datetime import datetime # Import datetime untuk nama file unik

# --- Inisialisasi Aplikasi dan Model ---

app = FastAPI(title="API Deteksi Objek YOLO")

# --- Memuat file .onnx yang lebih portabel ---
MODEL_PATH = os.path.join("model", "best.onnx")

# Folder untuk menyimpan gambar hasil deteksi
OUTPUT_IMAGE_DIR = "hasil_deteksi" 
# Buat folder jika belum ada
os.makedirs(OUTPUT_IMAGE_DIR, exist_ok=True) 

try:
    # --- Secara eksplisit menentukan tugas model untuk menghilangkan peringatan ---
    model = YOLO(MODEL_PATH, task='detect')
    print(f"Model berhasil dimuat dari: {MODEL_PATH}")
except Exception as e:
    print(f"Gagal memuat model: {e}")
    model = None

# --- API Endpoints ---

@app.get("/")
def read_root():
    return {"status": "ok", "pesan": "API YOLO berjalan dengan baik."}

@app.post("/detect/")
async def detect_objects(file: UploadFile = File(...)):
    if not model:
        raise HTTPException(status_code=500, detail="Model tidak dapat dimuat. Periksa log server.")

    contents = await file.read()
    
    try:
        img = Image.open(io.BytesIO(contents))
        
        # Lakukan deteksi objek
        results = model(img)

        # Proses hasil deteksi untuk JSON
        detections = []
        path_gambar_hasil = None # Inisialisasi path gambar hasil

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
                
                # --- Menyimpan gambar dengan boundary box ---
                
                # Buat nama file yang unik dengan timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_filename = f"{timestamp}_{file.filename}"
                path_gambar_hasil = os.path.join(OUTPUT_IMAGE_DIR, output_filename)
                
                # Gunakan metode .plot() untuk menggambar kotak dan simpan hasilnya
                annotated_image = result.plot() # Ini menghasilkan gambar format BGR (NumPy array)
                
                # Konversi dari BGR ke RGB dan simpan menggunakan Pillow
                annotated_image_rgb = Image.fromarray(annotated_image[...,::-1]) # Konversi BGR -> RGB
                annotated_image_rgb.save(path_gambar_hasil)
                # ----------------------------------------------------

        # Kembalikan JSON yang berisi data deteksi dan path gambar hasil
        return JSONResponse(content={
            "deteksi": detections,
            "gambar_hasil": path_gambar_hasil
        })

    except Exception as e:
        print(f"Terjadi error saat memproses gambar: {e}")
        raise HTTPException(status_code=500, detail=f"Gagal memproses gambar. Error: {e}")

@app.get("/status")
def get_status():
    return {"model_loaded": model is not None}
