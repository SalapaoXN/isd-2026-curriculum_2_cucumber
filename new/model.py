import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
from PIL import Image
from dotenv import load_dotenv
from preprocess_img import resize_if_needed
load_dotenv(r'C:\Users\TUF\OneDrive\Desktop\kmitl\ISD\new\.gitignore\.env')



class DocumentDataset(Dataset):
    def __init__(self, image_paths, max_size=1800):
        self.image_paths = image_paths
        self.max_size = max_size

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # โหลดรูปและแปลงเป็น RGB
        img = Image.open(self.image_paths[idx]).convert("RGB")
        img = resize_if_needed(img, self.max_size)
        return img, self.image_paths[idx]


class OCR_model:
    def __init__(self,bath_size = 2):
        self.bath_size = bath_size
    def model_load_toRAM(self):
        try:
            quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            )
            model_id = "typhoon-ai/typhoon-ocr1.5-2b"
            print('load model 4 bit')
            self.processor = AutoProcessor.from_pretrained(model_id)

            self.model = AutoModelForImageTextToText.from_pretrained(
                    model_id,
                    device_map="auto",
                    quantization_config=quantization_config,
            )
            return self
        except:
            return False

    def answer(self, dataset):
        # ฟังก์ชันช่วยจัดกลุ่มข้อมูลให้ DataLoader
        def custom_collate(batch):
            print(batch)
            images = [item[0] for item in batch]
            paths = [item[1] for item in batch]
            return images, paths
        # สร้าง DataLoader เพื่อรันข้อมูลทีละ Batch
        dataloader = DataLoader(dataset, batch_size=self.bath_size, shuffle=False, collate_fn=custom_collate , pin_memory=True)
        
        prompt = """Extract all text from the image. Instructions: - Only return the clean Markdown. - Do not include any explanation or extra text. - You must include all information on the page. Formatting Rules: - Tables: Render tables using <table>...</table> in clean HTML format. - Equations: Render equations using LaTeX syntax with inline ($...$) and block ($$...$$). - Images/Charts/Diagrams: Wrap any clearly defined visual areas (e.g. charts, diagrams, pictures) in: <figure> Describe the image's main elements (people, objects, text), note any contextual clues (place, event, culture), mention visible text and its meaning, provide deeper analysis when relevant (especially for financial charts, graphs, or documents), comment on style or architecture if relevant, then give a concise overall summary. Describe in Thai. </figure> - Page Numbers: Wrap page numbers in <page_number>...</page_number> (e.g., <page_number>14</page_number>). - Checkboxes: Use ☐ for unchecked and ☑ for checked boxes."""

        results = [] # เอาไว้เก็บผลลัพธ์ทั้งหมด
        
        for batch_imgs, batch_paths in dataloader:
            # สร้างข้อความสำหรับทุกรูปใน Batch
            batch_messages = []
            for img in batch_imgs:
                batch_messages.append([
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": img},
                            {"type": "text", "text": prompt}
                        ]
                    }
                ])
                
            inputs = self.processor.apply_chat_template(
                batch_messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt"
            )
            inputs = inputs.to(self.model.device)
            
            with torch.no_grad():
                generated_ids = self.model.generate(**inputs, max_new_tokens=2400)
                
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            
            output_texts = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            
            # จับคู่ที่อยู่ไฟล์กับข้อความที่แปลงได้ เก็บลง List
            for path, text in zip(batch_paths, output_texts):
                results.append({"path": path, "text": text})
                print(f"ประมวลผลเสร็จสิ้น: {path}")
        print(results)
        return results


    def load_in_to(self, folder_jpg_path):
        # 1. สร้าง List ว่างเพื่อเก็บที่อยู่ไฟล์
        image_paths = []
        
        # 2. วนลูปอ่านทุกไฟล์ในโฟลเดอร์ที่ระบุ
        for file_name in os.listdir(folder_jpg_path):
                # นำโฟลเดอร์ + ชื่อไฟล์ มาต่อกันให้เป็น Path ที่สมบูรณ์
                full_path = os.path.join(folder_jpg_path, file_name)
                image_paths.append(full_path)

        # 3. เช็กว่าเจอไฟล์รูปบ้างไหม
        if not image_paths:
            print(f" ไม่พบไฟล์รูปภาพในโฟลเดอร์ {folder_jpg_path}")
            return None
            
        # 4. ส่ง List ของ Path ทั้งหมดเข้าไปสร้าง Dataset
        return self.answer(DocumentDataset(image_paths))

