from PIL import Image
def resize_if_needed(img, max_size):
    width, height = img.size
    # เปลี่ยนเงื่อนไขเป็น: ทำงานก็ต่อเมื่อรูปใหญ่กว่า max_size (1800)
    if width > max_size or height > max_size: 
        if width >= height:
            scale = max_size / float(width)
            new_size = (max_size, int(height * scale))
        else:
            scale = max_size / float(height)
            new_size = (int(width * scale), max_size)
        img = img.resize(new_size, Image.Resampling.LANCZOS)
        print(f"\033[91mลดขนาดรูปภาพจาก {width}x{height} ==> {img.size}")
        return img
    else:
        print(f"\033[92mขนาดรูปภาพ {width}x{height} ไม่เกินที่กำหนด ไม่ต้องปรับขนาด\033[0m")
        return img

