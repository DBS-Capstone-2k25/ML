from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os
import traceback

# --- Konfigurasi ---
# Nama base model dari Hugging Face
BASE_MODEL_NAME = "Qwen/Qwen2-7B-Instruct"

# Path LOKAL di dalam container tempat adapter LoRA Anda berada.
# Path ini harus sesuai dengan yang ada di Dockerfile Anda (hasil dari 'COPY . .')
ADAPTER_PATH = "./qwen_chatbot_huggingface_space"

# --- Inisialisasi Aplikasi FastAPI ---
app = FastAPI(
    title="Qwen2 Chatbot API",
    description="API untuk berinteraksi dengan model Qwen2-7B yang disesuaikan dengan LoRA.",
    version="1.0.0",
)

# --- Variabel Global untuk Model dan Tokenizer ---
# Dibiarkan None pada awalnya, akan diisi saat startup
model = None
tokenizer = None

# --- Blok Pemuatan Model saat Aplikasi FastAPI Dimulai ---
# Ini adalah bagian paling kritis yang telah diperbaiki.
@app.on_event("startup")
def load_model():
    global model, tokenizer

    # Menggunakan try-except untuk menangkap error apa pun selama pemuatan
    # dan mencetaknya ke log Cloud Run untuk kemudahan debugging.
    try:
        print("--- Memulai proses pemuatan model untuk lingkungan CPU ---")

        # Langkah 1: Memastikan direktori adapter ada
        if not os.path.isdir(ADAPTER_PATH):
            print(f"❌ FATAL: Direktori adapter '{ADAPTER_PATH}' tidak ditemukan. Pastikan nama folder sudah benar dan tercopy ke dalam container.")
            raise RuntimeError(f"Adapter path not found: {ADAPTER_PATH}")
        else:
             print(f"✅ Direktori adapter '{ADAPTER_PATH}' ditemukan.")

        # Langkah 2: Memuat base model dengan konfigurasi untuk CPU
        # PERBAIKAN: Menghapus `load_in_4bit`, `torch_dtype`, dan mengubah `device_map` ke "cpu"
        print(f"Memuat base model '{BASE_MODEL_NAME}'...")
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_NAME,
            device_map="cpu",  # <-- WAJIB: Memaksa model untuk dimuat dan berjalan di CPU
        )
        print("✅ Base model berhasil dimuat.")

        # Langkah 3: Menerapkan adapter LoRA di atas base model
        print(f"Menerapkan adapter LoRA dari '{ADAPTER_PATH}'...")
        model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
        print("✅ Adapter LoRA berhasil diterapkan.")

        # Langkah 4: Memuat tokenizer dari direktori adapter
        print(f"Memuat tokenizer dari '{ADAPTER_PATH}'...")
        tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print("✅ Tokenizer berhasil dimuat.")

        print("🎉🎉🎉 Model dan tokenizer siap digunakan! 🎉🎉🎉")

    except Exception as e:
        # Jika terjadi error, catat traceback lengkap ke log untuk dianalisis
        print(f"❌ FATAL ERROR SAAT MEMUAT MODEL: {e}")
        print(traceback.format_exc())
        # Variabel model dibiarkan None, sehingga endpoint akan mengembalikan error 503
        model = None
        tokenizer = None

# --- Definisi Model Request Body ---
class ChatRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 300

# --- Endpoint Aplikasi ---
@app.get("/", summary="Cek Status API")
def read_root():
    """Endpoint untuk memeriksa apakah API sedang berjalan."""
    return {"status": "Qwen2 Chatbot API is running"}

@app.post("/chat", summary="Hasilkan Respons Chat")
async def generate_chat_response(request: ChatRequest):
    """
    Menerima prompt dari user dan menghasilkan respons dari model chatbot.
    """
    # Jika model gagal dimuat saat startup, kembalikan error 503 Service Unavailable
    if not model or not tokenizer:
        raise HTTPException(
            status_code=503,
            detail="Model is not available or failed to load. Check Cloud Run logs for startup errors."
        )

    try:
        # Membuat template chat yang sesuai untuk model Qwen2
        messages = [{"role": "user", "content": request.prompt}]
        prompt_template = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Tokenisasi input dan kirim ke perangkat tempat model berada (CPU)
        inputs = tokenizer(prompt_template, return_tensors="pt").to(model.device)

        # Hasilkan respons dari model
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=request.max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        # Decode output token menjadi teks
        response_ids = outputs[0][inputs.input_ids.shape[-1]:]
        response_text = tokenizer.decode(response_ids, skip_special_tokens=True).strip()

        return {"response": response_text}

    except Exception as e:
        # Menangkap error yang mungkin terjadi selama proses generasi
        print(f"Error selama proses generasi: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred during text generation: {str(e)}")
