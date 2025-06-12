# main.py

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import uvicorn
import os

# --- 1. Konfigurasi ---
# Path ini harus sesuai dengan struktur folder proyek Anda.
# Pastikan Anda menempatkan folder hasil training di direktori yang sama dengan file main.py ini.
BASE_MODEL_NAME = "Qwen/Qwen2-1.5B-Instruct"
ADAPTER_PATH = "./qwen_1.5B_chatbot_p100_final"  # GANTI NAMA INI SESUAI NAMA FOLDER ANDA

# --- 2. Inisialisasi Aplikasi FastAPI ---
app = FastAPI(
    title="Chatbot Sampah API",
    description="API untuk berinteraksi dengan model LLM yang di-fine-tune untuk pengetahuan tentang sampah.",
    version="1.0.0"
)

# --- 3. Pemuatan Model (Hanya dijalankan sekali saat startup) ---
# Variabel global untuk menampung model dan tokenizer agar bisa diakses oleh endpoint
model = None
tokenizer = None

@app.on_event("startup")
def load_model():
    """
    Fungsi ini akan dijalankan sekali saat aplikasi FastAPI dimulai.
    Tugasnya adalah memuat model dan tokenizer ke dalam memori.
    """
    global model, tokenizer
    
    # Menggunakan CPU karena ini adalah target deployment akhir kita
    device = "cpu"
    print(f"--- Memulai Pemuatan Model pada Perangkat: {device} ---")
    
    try:
        # Memuat tokenizer dari folder lokal
        print(f"Memuat tokenizer dari: {ADAPTER_PATH}...")
        tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH, trust_remote_code=True)
        
        # Memuat base model untuk CPU
        print(f"Memuat base model (CPU mode): {BASE_MODEL_NAME}...")
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_NAME,
            torch_dtype=torch.float32, # float32 adalah standar yang stabil untuk CPU
            trust_remote_code=True
        )
        
        # Menggabungkan base model dengan adapter LoRA
        print("Menggabungkan model dengan adapter LoRA...")
        model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
        
        # Pindahkan seluruh model ke CPU dan set ke mode evaluasi
        model.to(device)
        model.eval()
        
        print("\n✅ Model berhasil dimuat dan siap menerima permintaan.")

    except Exception as e:
        print(f"\n❌ FATAL ERROR: Gagal memuat model. Aplikasi tidak akan bisa melakukan inferensi.")
        print(f"Error: {e}")

# --- 4. Mendefinisikan Model Request Body ---
# Ini menentukan format JSON yang kita harapkan dari klien (misal: dari website Anda)
class ChatRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 512 # Nilai default jika tidak disediakan oleh klien

# --- 5. Membuat Endpoint API ---

@app.get("/", tags=["General"])
def read_root():
    """Endpoint root untuk mengecek apakah server API berjalan."""
    return {"status": "ok", "message": "Selamat datang di Chatbot Sampah API!"}

@app.post("/chat", tags=["Inference"])
async def handle_chat(request: ChatRequest):
    """
    Endpoint utama untuk melakukan inferensi. Menerima prompt dan mengembalikan respons.
    """
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model sedang tidak tersedia atau gagal dimuat. Silakan cek log server.")

    try:
        # Menyiapkan input untuk model sesuai dengan chat template
        messages = [{"role": "user", "content": request.prompt}]
        prompt_chat_template = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = tokenizer(prompt_chat_template, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=request.max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.95,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # --- LOGIKA EKSTRAKSI JAWABAN YANG SUDAH DIPERBAIKI ---
        
        # 1. Dapatkan panjang prompt input dalam bentuk token
        prompt_length = inputs.input_ids.shape[1]
        
        # 2. Ambil hanya token yang dihasilkan (setelah prompt)
        response_ids = outputs[0][prompt_length:]
        
        # 3. Decode hanya token jawaban menjadi teks
        assistant_response = tokenizer.decode(response_ids, skip_special_tokens=True).strip()
        
        # ----------------------------------------------------

        return {"response": assistant_response}

    except Exception as e:
        print(f"Error saat inferensi: {e}")
        raise HTTPException(status_code=500, detail="Terjadi kesalahan internal saat memproses permintaan Anda.")

# --- 6. Menjalankan Server (Untuk testing lokal) ---
if __name__ == "__main__":
    # Perintah ini akan menjalankan server di http://127.0.0.1:8000
    # Gunakan 'uvicorn main:app --reload' di terminal untuk pengembangan
    uvicorn.run(app, host="0.0.0.0", port=8000)