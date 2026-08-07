from pathlib import Path
from typing import List, Union
import easyocr


class OCREngine:
    def __init__(self, languages: List[str] = None, gpu: bool = True):
        """Initialize EasyOCR model once in memory."""
        if languages is None:
            languages = ['th', 'en']
            
        print(f"Initializing EasyOCR (Languages: {languages}, GPU: {gpu})...")
        self.reader = easyocr.Reader(languages, gpu=gpu)

    def extract_text(self, image_path: Union[str, Path], detail: int = 0) -> List:
        """Extract text from an image.
        
        detail=0 -> returns list of strings.
        detail=1 -> returns bounding boxes, text, and confidence scores.
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found at path: {image_path}")

        return self.reader.readtext(str(image_path), detail=detail)