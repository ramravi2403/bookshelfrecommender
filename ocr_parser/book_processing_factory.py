from ocr_parser.book_metadata_parser import BookMetadataParser
from ocr_parser.book_processing_service import BookProcessingService
from ocr_parser.dataset_exporter import CsvDatasetExporter
from ocr_parser.google_books_client import GoogleBooksClient
from ocr_parser.language_model_adapter import LangChainModelAdapter
from ocr_parser.storage import JsonCacheStorage


class BookProcessingFactory:

    @staticmethod
    def create_processing_service(
            model_name: str = "gemini-2.0-flash",
            prompt_template_path: str = "ocr_parser/parser_prompt.md",
            cache_file: str = None,
            batch_size: int = 10,
            batch_delay: float = 5.0
    ) -> BookProcessingService:

        if cache_file is None:
            model_prefix = model_name.replace(".", "_").replace("-", "_")
            cache_file = f"{model_prefix}_book_cache.json"
        language_model = LangChainModelAdapter(model_name, prompt_template_path)
        cache_storage = JsonCacheStorage(cache_file)
        metadata_parser = BookMetadataParser(language_model, cache_storage)
        book_api_client = GoogleBooksClient()
        dataset_exporter = CsvDatasetExporter()

        return BookProcessingService(
            metadata_parser=metadata_parser,
            book_api_client=book_api_client,
            dataset_exporter=dataset_exporter,
            batch_size=batch_size,
            batch_delay=batch_delay
        )