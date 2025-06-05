from ocr_parser.book_processing_factory import BookProcessingFactory


class BookOCRProcessor:
    def __init__(self, model_name: str = "gemini-2.0-flash", batch_size: int = 5):
        self.model_name = model_name
        self.batch_size = batch_size
        self.processing_service = BookProcessingFactory.create_processing_service(
            model_name=self.model_name,
            batch_size=self.batch_size
        )

    async def process(self, ocr_texts: list[str], output_file: str) -> None:
        await self.processing_service.process_ocr_texts_to_dataset(ocr_texts, output_file)