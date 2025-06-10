from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os

# --- Konfigurasi ---
# Nama base model yang Anda gunakan saat training
BASE_MODEL_NAME = "Qwen/Qwen2-7B-Instruct"
# Path ke folder yang berisi adapter LoRA dan tokenizer Anda
ADAPTER_PATH = "./qwen_chatbot_huggingface_space"

# --- Inisialisasi Aplikasi FastAPI ---
app = FastAPI()

# --- Memuat Model dan Tokenizer saat Aplikasi Dimulai ---
# Ini akan memakan waktu dan memori, dan hanya dijalankan sekali.
try:
    print("Memuat base model...")
    # Muat base model dalam 4-bit untuk efisiensi memori
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        load_in_4bit=True,
        torch_dtype=torch.float16,
        device_map="auto",  # Otomatis menempatkan model di GPU jika tersedia
    )
    print("Base model dimuat.")

    print(f"Memuat adapter LoRA dari {ADAPTER_PATH}...")
    # Gabungkan base model dengan adapter LoRA Anda
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
    # Jika model gagal dimuat, aplikasi tidak dapat berjalan
    model = None
    tokenizer = None

# --- Mendefinisikan Format Input API ---
class ChatRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 300 # Memberi default value

# --- Endpoint untuk Health Check ---
@app.get("/")
def read_root():
    return {"status": "Model API is running"}

# --- Endpoint Utama untuk Chat ---
@app.post("/chat")
async def generate_chat_response(request: ChatRequest):
    if not model or not tokenizer:
        raise HTTPException(status_code=503, detail="Model is not available or failed to load.")

    try:
        # Format input sesuai dengan chat template Qwen2
        messages = [{"role": "user", "content": request.prompt}]
        prompt_template = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Tokenisasi input
        inputs = tokenizer(prompt_template, return_tensors="pt").to(model.device)

        # Generate respons dari model
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

        # Decode output menjadi teks
        response_ids = outputs[0][inputs.input_ids.shape[-1]:]
        response_text = tokenizer.decode(response_ids, skip_special_tokens=True).strip()

        return {"response": response_text}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during generation: {e}")
