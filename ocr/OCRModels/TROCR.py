#import torchvision

from ocr.OCRModels import OCRModel
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import torch
torch.set_num_threads(1)
from PIL import Image


class TROCR(OCRModel):
    def __init__(self, model_name: str = "microsoft/trocr-base-printed"):
        super().__init__("TROCR")
        try:

            self.processor = TrOCRProcessor.from_pretrained(model_name)
            self.model = VisionEncoderDecoderModel.from_pretrained(model_name)
            self.device = torch.device("cuda" if torch.cuda.is_available() else "mps")
            self.model.to(self.device)
        except ImportError:
            print("Warning: transformers package not found. Install with 'pip install transformers'")

    def detect_text(self, image_path: str) -> str:
        try:
            image = Image.open(image_path).convert("RGB")
            pixel_values = self.processor(image, return_tensors="pt").pixel_values.to(self.device)
            generated_ids = self.model.generate(pixel_values)
            raw_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            return self.normalize_text(raw_text)
        except Exception as e:
            print(f"TrOCR Error: {e}")
            return ""
