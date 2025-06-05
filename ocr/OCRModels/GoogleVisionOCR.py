import os

import cv2
import numpy as np

import ocr.Credentials
from ocr import Credentials
from ocr.OCRModels.OCRModel import OCRModel
from google.cloud import vision
from google.oauth2 import service_account
import io


class GoogleVisionOCR(OCRModel):
    def __init__(self, credentials_path: str = None):
        super().__init__('GoogleVision')
        if credentials_path is None:
            credentials_path = Credentials.GOOGLE_APPLICATION_CREDENTIALS
        if credentials_path and os.path.exists(credentials_path):
            os.environ[
                'GOOGLE_APPLICATION_CREDENTIALS'] = Credentials.GOOGLE_APPLICATION_CREDENTIALS

            self.client = vision.ImageAnnotatorClient()
        else:
            self.client = vision.ImageAnnotatorClient()

    def detect_text_from_array(self, image_array: np.ndarray) -> str:
        try:
            rgb_image = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
            success, encoded_image = cv2.imencode('.png', rgb_image)
            if not success:
                raise ValueError("Failed to encode image")

            content = encoded_image.tobytes()
            image = vision.Image(content=content)
            return self._process_vision_request(image)
        except Exception as e:
            print(f"Google Vision Error processing array: {e}")
            return ""

    def _process_vision_request(self, image: vision.Image) -> str:

        response = self.client.text_detection(image=image)
        texts = response.text_annotations

        if texts:
            raw_text = texts[0].description
            return self.normalize_text(raw_text)

        if response.error.message:
            print(f"Google Vision Error: {response.error.message}")
            return ""

        return ""

    def detect_text(self, image_path: str) -> str:

        try:

            with io.open(image_path, 'rb') as image_file:
                content = image_file.read()
            image = vision.Image(content=content)
            response = self.client._process_vision_request(image=image)
            texts = response.text_annotations
            if texts:
                raw_text = texts[0].description
                return self.normalize_text(raw_text)
            if response.error.message:
                print(f"Google Vision Error: {response.error.message}")
                return ""
            return ""
        except Exception as e:
            print(f"Google Vision Error: {e}")
            return ""
