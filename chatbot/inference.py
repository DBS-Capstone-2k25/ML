import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# --- KONFIGURASI (Ubah Sesuai Kebutuhan Anda) ---

# 1. Nama base model yang Anda gunakan saat training.
BASE_MODEL_NAME = "Qwen/Qwen2-1.5B-Instruct"

# 2. Path LOKAL ke folder yang berisi adapter LoRA dan tokenizer Anda.
#    PASTIKAN NAMA FOLDER INI SESUAI DENGAN HASIL DOWNLOAD ANDA.
ADAPTER_PATH = "./qwen_1.5B_chatbot_p100_final" 

def run_chat_test():
    """
    Fungsi untuk memuat model yang sudah di-fine-tune dan memulai sesi chat interaktif.
    Dioptimalkan untuk GPU NVIDIA RTX 30-series (Ampere).
    """
    print("--- Memulai Sesi Tes Inferensi untuk RTX 3050 ---")

    if not torch.cuda.is_available():
        print("Peringatan: GPU tidak terdeteksi. Skrip ini dioptimalkan untuk GPU.")
        # Anda bisa memilih untuk berhenti atau melanjutkan dengan CPU yang sangat lambat.
        # return 
    
    print(f"Menggunakan perangkat: {torch.cuda.get_device_name(0)}")

    # --- Pemuatan Model dan Tokenizer ---
    try:
        # Langkah 1: Muat tokenizer dari path lokal Anda.
        print(f"Memuat tokenizer dari: {ADAPTER_PATH}...")
        tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH, trust_remote_code=True)
        
        # OPSI WAJIB untuk RTX 3050: Gunakan kuantisasi 4-bit dan bfloat16.
        # - `load_in_4bit=True` wajib karena VRAM laptop yang terbatas.
        # - `torch_dtype=torch.bfloat16` adalah tipe data yang optimal untuk arsitektur Ampere (RTX 30-series).
        print(f"Memuat base model (4-bit, bfloat16 optimized) dari: {BASE_MODEL_NAME}...")
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_NAME,
            load_in_4bit=True,
            torch_dtype=torch.bfloat16,
            device_map="auto", # Biarkan transformers menangani penempatan di GPU
            trust_remote_code=True
        )
        
        print("Menggabungkan base model dengan adapter LoRA...")
        model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
        
        # Set model ke mode evaluasi
        model.eval()
        
        print("\nModel dan tokenizer berhasil dimuat. Siap untuk chat!")

    except Exception as e:
        print(f"\nTerjadi error saat memuat model: {e}")
        print("Pastikan nama model dan path adapter sudah benar, dan library (termasuk bitsandbytes) sudah terinstall.")
        return

    # --- Sesi Chat Interaktif ---
    messages_history = []
    print("\n--- Chatbot Siap ---")
    print("Ketik pertanyaan Anda. Ketik 'quit' atau 'exit' untuk keluar.")
    
    while True:
        user_input = input("Anda: ")
        if user_input.lower() in ["quit", "exit"]:
            break

        messages_history.append({"role": "user", "content": user_input})
        
        prompt_chat_template = tokenizer.apply_chat_template(
            messages_history,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = tokenizer(prompt_chat_template, return_tensors="pt").to(model.device)

        print("Chatbot sedang berpikir...")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7,
                top_p=0.95,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response_ids = outputs[0][inputs.input_ids.shape[-1]:]
        response_text = tokenizer.decode(response_ids, skip_special_tokens=True).strip()
        
        print(f"Chatbot: {response_text}")

        messages_history.append({"role": "assistant", "content": response_text})
        
        if len(messages_history) > 10:
            messages_history = messages_history[-10:]

    print("\n--- Sesi chat selesai. ---")


if __name__ == "__main__":
    run_chat_test()