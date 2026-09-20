from pathlib import Path
from typing import Any, Dict, List, Union
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

        results = self.reader.readtext(str(image_path), detail=detail)
        if detail == 0:
            return [text.upper() for text in results]

        if detail == 1:
            return [self._normalize_detection(result) for result in results]

        return results

    @classmethod
    def _normalize_detection(cls, result: Any) -> Dict[str, Any]:
        """Normalize EasyOCR detail-1 output without changing its order."""
        bbox, text, confidence = result
        return {
            "text": str(cls._to_builtin(text)),
            "confidence": cls._to_builtin(confidence),
            "bbox": cls._to_builtin(bbox),
        }

    @classmethod
    def _to_builtin(cls, value: Any) -> Any:
        """Convert numpy-like EasyOCR values into JSON-safe Python values."""
        if hasattr(value, "tolist"):
            value = value.tolist()
        elif hasattr(value, "item"):
            value = value.item()
        if isinstance(value, (list, tuple)):
            return [cls._to_builtin(item) for item in value]
        if isinstance(value, dict):
            return {str(key): cls._to_builtin(item) for key, item in value.items()}
        return value
