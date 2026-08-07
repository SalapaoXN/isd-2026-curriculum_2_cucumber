import easyocr
from utils.io import ensure_dir, save_json, save_text

def ocr(path):
    reader = easyocr.Reader(['th', 'en'], gpu=True)

    result = reader.readtext(path, detail=0)

    return result

def save_output(data, output_dir, name):
    save_json(data.to_dict(), output_dir / f"{name}_ocr.json")
    extracted_text = "\n".join(data)
    save_text(extracted_text, output_dir / f"{name}_ocr.txt")

def main():
    path = 'inputs/dsba/curriculum_page_016.jpg'
    result = ocr(path)
    result_dict = {
        "page" : result[0]
    }
    print(result, type(result))
    save_output(result, "outputs", path[-13:-3])
main()