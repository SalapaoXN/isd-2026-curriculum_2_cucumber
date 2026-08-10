'''Extract file pdf to multi png file'''

import pymupdf

doc = pymupdf.open(r"D:\dataset\หลักสูตรวิทยาศาสตรบัณฑิต สาขาวิชาวิทยาการข้อมูลและการวิเคราะห์เชิงธุรกิจ หลักสูตรปรับปรุง (พ.ศ. 2565).pdf")

for i in range(22,36):
    
    page = doc[i]
    
    # 1. ดึงขนาดความกว้างและสูงของหน้าปัจจุบัน
    page_width = page.rect.width
    page_height = page.rect.height
    
    # 2. กำหนดพิกัดที่ต้องการครอป (ตัวอย่าง: ตัดขอบบน ขอบล่าง ขอบซ้าย ขอบขวา ออกอย่างละ 10%)
    # รูปแบบ: pymupdf.Rect( x_เริ่ม , y_เริ่ม , x_สิ้นสุด , y_สิ้นสุด )
    crop_rect = pymupdf.Rect(
        page_width * 0.01,   # เริ่มจาก 1% ของความกว้างด้านซ้าย
        page_height * 0.10,  # เริ่มจาก 10% ของความสูงด้านบน
        page_width * 0.99,   # สิ้นสุดที่ 90% ของความกว้าง
        page_height * 0.80   # สิ้นสุดที่ 80% ของความสูง
    )
    
    # 3. สั่ง get_pixmap โดยใส่ clip=พิกัดที่ต้องการ
    pix = page.get_pixmap(dpi=150, clip=crop_rect)
    pix.pil_save(rf"D:\dataset\png_folder\page{i}.png")

doc.close()
