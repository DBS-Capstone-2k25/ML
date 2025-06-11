from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import uvicorn
from peft import PeftModel, PeftConfig # Tambahkan import ini

app = FastAPI()

# Definisikan path ke direktori model Anda
# Ini adalah direktori tempat semua file yang Anda unggah (adapter_config.json, dll.)
# dan juga bobot LoRA Anda (biasanya adapter_model.bin) disimpan.
# Asumsikan bobot model dasar Qwen2-7B-Instruct tidak ada di sini,
# tetapi akan diunduh oleh transformers jika tidak ada.
model_dir = "qwen_chatbot_huggingface_space" # Ganti dengan path ke direktori model LoRA Anda

try:
    # Muat konfigurasi PEFT dari adapter_config.json
    peft_config = PeftConfig.from_pretrained(model_dir)
    print(f"PEFT config loaded. Base model: {peft_config.base_model_name_or_path}")

    # Muat tokenizer dari base model
    # (tokenizer LoRA biasanya tidak berubah dari base model)
    tokenizer = AutoTokenizer.from_pretrained(peft_config.base_model_name_or_path)

    # Muat model dasar (misalnya "Qwen/Qwen2-7B-Instruct")
    # Pastikan untuk menentukan trust_remote_code=True jika diperlukan oleh model
    model = AutoModelForCausalLM.from_pretrained(
        peft_config.base_model_name_or_path,
        torch_dtype=torch.bfloat16, # Gunakan bfloat16 jika GPU mendukung, atau float16 untuk efisiensi
        # load_in_4bit=True, # Pertimbangkan ini jika memori GPU terbatas (membutuhkan bitsandbytes)
        # device_map="auto", # Memuat model ke perangkat yang tersedia secara otomatis
        trust_remote_code=True # Qwen sering membutuhkan ini
    )
    print("Base model loaded.")

    # Muat adapter LoRA ke model dasar
    model = PeftModel.from_pretrained(model, model_dir)
    print("LoRA adapter loaded and merged into base model.")

    # Set model ke mode evaluasi (penting untuk inferensi)
    model.eval()

except Exception as e:
    print(f"Error loading model with LoRA: {e}")
    # Jika Anda mengalami masalah, coba debug di sini
    raise e # Re-raise error untuk melihat stack trace lengkap

# Pindahkan model ke GPU jika tersedia dan belum ditangani oleh device_map="auto"
if torch.cuda.is_available():
    # Jika Anda menggunakan device_map="auto", model mungkin sudah di GPU.
    # Jika tidak, pindahkan secara manual.
    if model.device.type == 'cpu':
        model.to("cuda")
        print("Model moved to GPU manually.")
else:
    print("GPU not available, using CPU. Inference might be slow.")

class ChatRequest(BaseModel):
    message: str
    max_new_tokens: int = 50 # Meningkatkan default untuk jawaban yang lebih panjang
    temperature: float = 0.7
    top_p: float = 0.9 # Tambahkan top_p untuk kontrol sampling yang lebih baik

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        # Template chat dari tokenizer_config.json:
        # "{{ '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n' }}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"

        messages = [
            # Anda bisa menambahkan pesan sistem di sini jika diinginkan
            # {"role": "system", "content": "You are a helpful and friendly chatbot."},
            {"role": "user", "content": request.message}
        ]
        # Pastikan add_generation_prompt=True untuk model instruksi
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        with torch.no_grad(): # Nonaktifkan perhitungan gradien untuk inferensi
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=request.max_new_tokens,
                do_sample=True,
                temperature=request.temperature,
                top_p=request.top_p,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id # Penting untuk menghentikan generasi
            )

        # Qwen-2 sering menghasilkan ulang prompt, jadi kita perlu memotongnya
        # Perhatikan bahwa generated_ids sudah dipotong sebelumnya di contoh sebelumnya.
        # Jika Anda ingin memotong yang spesifik untuk Qwen2, bisa seperti ini:
        response_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        # Hapus bagian prompt dari respons jika model mengulanginya
        # Cari posisi akhir dari prompt dalam teks yang dihasilkan
        prompt_end_index = response_text.find(text)
        if prompt_end_index != -1:
            response_text = response_text[prompt_end_index + len(text):].strip()
        
        # Tambahkan pembersihan token spesial yang mungkin muncul di akhir
        # Ini opsional, tapi bisa membantu jika ada sisa token spesial
        for special_token in tokenizer.all_special_tokens:
            response_text = response_text.replace(special_token, "").strip()


        return {"response": response_text}
    except Exception as e:
        print(f"Error during chat generation: {e}")
        return {"error": str(e)}

# Jalankan aplikasi dengan Uvicorn (untuk pengembangan)
if __name__ == "__main__":
    # Jika Anda ingin mengizinkan akses dari jaringan lain di lokal (misalnya dari perangkat lain di LAN)
    # Anda bisa menggunakan host="0.0.0.0". Untuk pengujian hanya dari komputer ini,
    # host="127.0.0.1" atau host="localhost" sudah cukup.
    uvicorn.run(app, host="0.0.0.0", port=8000)