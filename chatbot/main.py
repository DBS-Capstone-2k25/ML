from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os
# --- Konfigurasi ---
BASE_MODEL_NAME = "Qwen/Qwen2-7B-Instruct"
# Pastikan path ini sesuai dengan nama folder Anda
ADAPTER_PATH = "./qwen_chatbot_huggingface_space"

# --- Inisialisasi Aplikasi FastAPI ---
app = FastAPI()

# --- Global variables untuk model dan tokenizer ---
model = None
tokenizer = None

# --- Blok Pemuatan Model saat Startup ---
try:
    # --- BLOK DEBUGGING UNTUK MEMASTIKAN FILE ADA ---
    print("--- Memulai proses debugging path ---")
    print(f"Mencari direktori di path: {ADAPTER_PATH}")
    if os.path.isdir(ADAPTER_PATH):
        print(f"✅ SUKSES: Direktori '{ADAPTER_PATH}' ditemukan.")
        try:
            file_list = os.listdir(ADAPTER_PATH)
            print(f"Isi direktori: {file_list}")
            # Cek keberadaan file krusial
            if "adapter_model.safetensors" in file_list and "tokenizer.json" in file_list:
                print("✅ SUKSES: File 'adapter_model.safetensors' dan 'tokenizer.json' terdeteksi.")
            else:
                print("❌ PERINGATAN: File model atau tokenizer krusial tidak ditemukan di dalam direktori!")
        except Exception as e:
            print(f"❌ ERROR: Tidak bisa membaca isi direktori: {e}")
    else:
        print(f"❌ FATAL: Direktori '{ADAPTER_PATH}' TIDAK DITEMUKAN.")
        # Kita bisa hentikan proses lebih awal jika mau, tapi kita biarkan lanjut agar error utama tertangkap
    print("--- Selesai proses debugging path ---")
    # ---------------------------------------------------

    print("Memuat base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME
    )
    print("Base model dimuat.")

    print(f"Memuat adapter LoRA dari {ADAPTER_PATH}...")
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    print("Adapter LoRA dimuat dan digabungkan.")

    print(f"Memuat tokenizer dari {ADAPTER_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("Tokenizer dimuat.")

    print("Model dan tokenizer siap digunakan.")

except Exception as e:
    print(f"Error saat memuat model: {e}")
    model = None
    tokenizer = None

# --- Definisi Endpoint (tetap sama seperti sebelumnya) ---
class ChatRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 300

@app.get("/")
def read_root():
    return {"status": "Model API is running"}

@app.post("/chat")
async def generate_chat_response(request: ChatRequest):
    if not model or not tokenizer:
        # Pesan ini akan muncul jika blok try-except di atas gagal
        raise HTTPException(status_code=503, detail="Model is not available or failed to load. Check logs for details.")
    # ... sisa kode endpoint tetap sama ...
    try:
        messages = [{"role": "user", "content": request.prompt}]
        prompt_template = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = tokenizer(prompt_template, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=request.max_new_tokens,
                do_sample=True,
                temperature=0.6,
                top_p=0.95,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        response_ids = outputs[0][inputs.input_ids.shape[-1]:]
        response_text = tokenizer.decode(response_ids, skip_special_tokens=True).strip()
        return {"response": response_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during generation: {e}")
