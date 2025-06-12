import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# --- KONFIGURASI (Ubah Sesuai Kebutuhan Anda) ---

# 1. Nama base model yang Anda gunakan saat training.
BASE_MODEL_NAME = "Qwen/Qwen2-1.5B-Instruct"

# 2. Path LOKAL ke folder yang berisi adapter LoRA dan tokenizer Anda.
ADAPTER_PATH = "./qwen_1.5B_chatbot_p100_final" # PASTIKAN NAMA FOLDER INI BENAR

def run_chat_test():
    """
    Fungsi untuk memuat model dan menjalankan inferensi HANYA MENGGUNAKAN CPU.
    """
    print("--- Memulai Sesi Tes Inferensi (CPU Mode) ---")

    # Secara eksplisit paksa penggunaan CPU
    device = "cpu"
    print(f"Menggunakan perangkat: {device}")

    # --- Pemuatan Model dan Tokenizer ---
    try:
        # Langkah 1: Muat tokenizer dari path lokal.
        print(f"Memuat tokenizer dari: {ADAPTER_PATH}...")
        tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH, trust_remote_code=True)
        
        # Langkah 2: Muat base model untuk CPU.
        # Tidak ada `load_in_4bit=True`. Menggunakan presisi standar float32.
        # Pemuatan pertama kali akan mengunduh model ~3GB.
        print(f"Memuat base model (CPU mode) dari: {BASE_MODEL_NAME}...")
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_NAME,
            torch_dtype=torch.float32,  # float32 adalah standar yang stabil untuk CPU
            trust_remote_code=True
        ).to(device) # Pindahkan model ke CPU
        
        print("Menggabungkan base model dengan adapter LoRA...")
        model = PeftModel.from_pretrained(base_model, ADAPTER_PATH).to(device) # Pindahkan juga model gabungan ke CPU
        
        model.eval()
        
        print("\n✅ Model dan tokenizer berhasil dimuat. Siap untuk chat!")

    except Exception as e:
        print(f"\n❌ Terjadi error saat memuat model: {e}")
        print("Pastikan nama model dan path adapter sudah benar.")
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
            messages_history, tokenize=False, add_generation_prompt=True
        )

        inputs = tokenizer(prompt_chat_template, return_tensors="pt").to(device)

        print("Chatbot sedang berpikir (CPU, mungkin agak lama)...")
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