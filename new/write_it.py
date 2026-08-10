from model import OCR_model
import os
ocr = OCR_model(bath_size=4)
ocr.model_load_toRAM()
ans = ocr.load_in_to(r"D:\dataset\png_folder")

if ans is not None:
    with open('output.txt', mode='w', encoding='utf-8') as f:
        for item in ans:
            f.write(f"= Page: {os.path.basename(item['path'])} =\n")
            f.write(item['text'])
            f.write('\n\n')