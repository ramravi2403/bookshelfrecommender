from OCRModels.OCRModel import OCRModel
import pytesseract
from PIL import Image


class TesseractOCR(OCRModel):

    def __init__(self, lang: str = 'eng', config: str = ''):
        super().__init__('Tesseract')

        try:
            self.pytesseract = pytesseract
            self.lang = lang
            self.config = config
        except ImportError:
            print("Warning: pytesseract package not found. Install with 'pip install pytesseract'")
            print("You also need to install Tesseract OCR engine: https://github.com/tesseract-ocr/tesseract")

    def detect_text(self, image_path: str) -> str:

        try:

            image = Image.open(image_path)
            raw_text = self.pytesseract.image_to_string(image, lang=self.lang, config=self.config)

            return self.normalize_text(raw_text)
        except Exception as e:
            print(f"Tesseract OCR Error: {e}")
            return ""