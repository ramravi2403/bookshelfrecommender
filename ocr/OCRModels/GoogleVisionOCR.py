import os

import Credentials
from OCRModels.OCRModel import OCRModel
from google.cloud import vision
from google.oauth2 import service_account
import io


class GoogleVisionOCR(OCRModel):
    """Google Cloud Vision OCR implementation"""

    def __init__(self, credentials_path: str = None):
        super().__init__('GoogleVision')
        if credentials_path is None:
            credentials_path = Credentials.GOOGLE_APPLICATION_CREDENTIALS
        print(credentials_path)

        if credentials_path and os.path.exists(credentials_path):
            print("Im here")
            os.environ[
                'GOOGLE_APPLICATION_CREDENTIALS'] = Credentials.GOOGLE_APPLICATION_CREDENTIALS

            self.client = vision.ImageAnnotatorClient()
        else:
            self.client = vision.ImageAnnotatorClient()

    def detect_text(self, image_path: str) -> str:

        try:


            # Read the image file
            with io.open(image_path, 'rb') as image_file:
                content = image_file.read()

            image = vision.Image(content=content)

            # Perform text detection
            response = self.client.text_detection(image=image)
            texts = response.text_annotations

            # Check if there's any text detected
            if texts:
                # The first element contains the entire detected text
                raw_text = texts[0].description
                return self.normalize_text(raw_text)

            # Check for errors
            if response.error.message:
                print(f"Google Vision Error: {response.error.message}")
                return ""

            return ""
        except Exception as e:
            print(f"Google Vision Error: {e}")
            return ""
