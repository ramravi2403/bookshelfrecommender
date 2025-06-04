from abc import ABC, abstractmethod


class OCRModel(ABC):

    def __init__(self, model_name):
        self.__model_name = model_name

    @abstractmethod
    def detect_text(self, image_path: str) -> str:
        pass

    def name(self) -> str:
        return self.__model_name

    def normalize_text(self, text: str) -> str:
        return ' '.join(text.replace('\n', ' ').split()).lower()
