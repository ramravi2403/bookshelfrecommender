from abc import ABC, abstractmethod
from typing import List, Dict, Optional
import asyncio

from ocr_parser.models import DatasetRecord, BookInfo


class ILanguageModel(ABC):

    @abstractmethod
    async def generate_async(self, prompt: str, **kwargs) -> str:
        pass


class ICacheStorage(ABC):
    @abstractmethod
    def load(self) -> Dict[str, Dict]:
        pass

    @abstractmethod
    def save(self, data: Dict[str, Dict]) -> None:
        pass

    @abstractmethod
    def get(self, key: str) -> Optional[Dict]:
        pass

    @abstractmethod
    def set(self, key: str, value: Dict) -> None:
        pass


class IBookApiClient(ABC):
    @abstractmethod
    async def search_book(self, title: str, author: str) -> Optional[BookInfo]:
        pass


class IDatasetExporter(ABC):
    @abstractmethod
    def export(self, records: List[DatasetRecord], output_path: str) -> None:
        pass

