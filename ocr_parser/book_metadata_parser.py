import json
from typing import List, Optional, Dict
import asyncio

from ocr_parser.interfaces import ILanguageModel, ICacheStorage
from ocr_parser.models import BookMetadata


class BookMetadataParser:
    def __init__(self, language_model: ILanguageModel, cache_storage: ICacheStorage):
        self.language_model = language_model
        self.cache_storage = cache_storage

    async def parse_single_text(self, ocr_text: str) -> BookMetadata:
        cached_result = self.cache_storage.get(ocr_text)
        if cached_result:
            return BookMetadata(**cached_result)

        try:
            response = await self.language_model.generate_async(ocr_text)
            json_data = self._extract_json_from_response(response)
            self._validate_response(json_data)
            json_data['isbn13'] = str(json_data['isbn13'])
            result = BookMetadata(**json_data)

        except (json.JSONDecodeError, ValueError, KeyError) as e:
            print(f"Error processing text '{ocr_text}': {str(e)}")
            result = BookMetadata(title="NA", author="NA", isbn13="0")

        self.cache_storage.set(ocr_text, {
            'title': result.title,
            'author': result.author,
            'isbn13': result.isbn13
        })

        return result

    async def parse_multiple_texts(self, ocr_texts: List[str]) -> List[BookMetadata]:
        tasks = [self.parse_single_text(text) for text in ocr_texts]
        results = await asyncio.gather(*tasks)
        cache_data = self.cache_storage.load()
        self.cache_storage.save(cache_data)

        return results

    def _extract_json_from_response(self, response: str) -> Dict:
        start_idx = response.find('{')
        end_idx = response.rfind('}') + 1

        if start_idx == -1 or end_idx == 0:
            raise ValueError("No JSON object found in response")

        json_str = response[start_idx:end_idx]
        return json.loads(json_str)

    def _validate_response(self, json_data: Dict) -> None:
        required_fields = ['title', 'author', 'isbn13']
        if not all(field in json_data for field in required_fields):
            raise ValueError("Missing required fields in response")