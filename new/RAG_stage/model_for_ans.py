import torch

# ตัวที่จำเป็นจริง — ปิด cuDNN attention backend, เปิด backend อื่นแทน
torch.backends.cuda.enable_cudnn_sdp(False)
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(True)

# ปิด cuDNN สำหรับ Conv ด้วย เผื่อโมเดลมี Conv layer ปนอยู่บ้าง (เช่น embedding บาง arch)
torch.backends.cudnn.enabled = False

from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def test_typhoon_inference(retrieved_context , user_question):
    # 1. โหลด .env
    load_dotenv(r'C:\Users\TUF\OneDrive\Desktop\kmitl\ISD\new\.gitignore\.env')
    
    # 2. กำหนด Path โมเดล
    # **สำคัญ:** 
    # - ถ้าตอนโหลดคุณใช้ "วิธีที่ 1" (รันเป็น Admin) ให้ใช้: model_path = "scb10x/typhoon2.5-qwen3-4b"
    # - ถ้าตอนโหลดคุณใช้ "วิธีที่ 2" (ระบุ local_dir) ให้ใช้ Path โฟลเดอร์ตรงๆ ด้านล่างนี้:
    model_path = "scb10x/typhoon2.5-qwen3-4b"

    bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    )

    print("กำลังโหลด Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    print(f"กำลังโหลดโมเดลเข้า VRAM (bfloat16)...")
    # โหลดโมเดลด้วย bfloat16 เพื่อความรวดเร็วและประหยัด VRAM กว่าแบบ 32-bit ครึ่งหนึ่ง
    model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=bnb_config,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="eager"   # หรือ "sdpa" ถ้าอยาก perf ดีกว่าแต่ยังกัน cudnn_sdp
    )
    
    print("✅ โหลด Typhoon 2.5 (4B) พร้อมใช้งานแล้ว!\n")

    # 3. จำลองข้อความที่สกัดได้จาก Typhoon OCR
    print(f"ข้อความ Input จาก OCR: {user_question}")

    # 4. สร้าง Prompt 
    # (Typhoon2.5 เก่งภาษาไทยมาก เราสามารถสั่งงานเป็นภาษาไทยได้เลย)


    messages = [
    {
        "role": "system",
        "content": (
            "คุณคือผู้ช่วยตอบคำถามโดยอ้างอิงจากข้อมูลที่ให้มาเท่านั้น (Context) "
            "ห้ามตอบจากความรู้ทั่วไปหรือแต่งข้อมูลขึ้นเอง "
            "ถ้าไม่พบคำตอบในข้อมูลที่ให้ ให้ตอบว่า 'ไม่พบข้อมูลที่เกี่ยวข้องในเอกสาร' "
            "ตอบให้กระชับ ตรงประเด็น และอ้างอิงแหล่งที่มาถ้ามีระบุไว้"
        )
    },
    {
        "role": "user",
        "content": (
            f"ข้อมูลอ้างอิง (Context):\n{retrieved_context}\n\n"
            f"คำถาม: {user_question}"
        )
    }
]

    # แปลง Prompt ให้อยู่ในฟอร์แมตแชท
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    print("\nกำลังประมวลผลคำตอบ...")
    print(model_inputs)
    # 5. สั่งโมเดล Generate ข้อความ (ปรับ max_new_tokens ได้ตามความยาวที่ต้องการ)
    generated_ids = model.generate(**model_inputs, max_new_tokens=256)
    
    # ตัดส่วนที่เป็นคำถาม (Prompt) ออก เอาแค่ส่วนที่โมเดลตอบ
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    
    # ถอดรหัส Output กลับเป็นข้อความ
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    print("\n================ ผลลัพธ์ ================")
    print(response)
    print("=========================================")

