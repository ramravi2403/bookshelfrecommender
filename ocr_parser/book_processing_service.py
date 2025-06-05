import asyncio
from typing import List, Optional
from tqdm import tqdm

from ocr_parser.book_metadata_parser import BookMetadataParser
from ocr_parser.interfaces import IBookApiClient, IDatasetExporter
from ocr_parser.models import BookMetadata, DatasetRecord


class BookProcessingService:

    def __init__(
            self,
            metadata_parser: BookMetadataParser,
            book_api_client: IBookApiClient,
            dataset_exporter: IDatasetExporter,
            batch_size: int = 10,
            batch_delay: float = 5.0
    ):
        self.metadata_parser = metadata_parser
        self.book_api_client = book_api_client
        self.dataset_exporter = dataset_exporter
        self.batch_size = batch_size
        self.batch_delay = batch_delay

    async def process_ocr_texts_to_dataset(self, ocr_texts: List[str], output_path: str) -> None:
        print(f"Processing {len(ocr_texts)} OCR texts...")
        print("Step 1: Parsing OCR texts to structured metadata...")
        metadata_list = await self.metadata_parser.parse_multiple_texts(ocr_texts)
        print("Step 2: Enriching with Google Books API data...")
        dataset_records = await self._enrich_with_book_info(metadata_list)
        print("Step 3: Exporting dataset...")
        self.dataset_exporter.export(dataset_records, output_path)

    async def _enrich_with_book_info(self, metadata_list: List[BookMetadata]) -> List[DatasetRecord]:
        records = []
        total_batches = (len(metadata_list) + self.batch_size - 1) // self.batch_size

        with tqdm(total=len(metadata_list), desc="Enriching book data") as pbar:
            for i in range(0, len(metadata_list), self.batch_size):
                batch = metadata_list[i:i + self.batch_size]
                batch_records = await self._process_batch(batch)
                records.extend(batch_records)
                pbar.update(len(batch))
                if i + self.batch_size < len(metadata_list):
                    await asyncio.sleep(self.batch_delay)

        return records

    async def _process_batch(self, metadata_batch: List[BookMetadata]) -> List[DatasetRecord]:
        tasks = [self._process_single_metadata(metadata) for metadata in metadata_batch]
        results = await asyncio.gather(*tasks)
        return [record for record in results if record is not None]

    async def _process_single_metadata(self, metadata: BookMetadata) -> Optional[DatasetRecord]:
        if not metadata.is_valid():
            return None

        try:
            book_info = await self.book_api_client.search_book(metadata.title, metadata.author)

            if book_info is None:
                print(f"No book info found for: {metadata.title} by {metadata.author}")
                return None

            return DatasetRecord(
                title=metadata.title,
                author=metadata.author,
                genres=', '.join(book_info.genres) if book_info.genres != ["NA"] else '',
                rating=book_info.average_rating,
                rating_count=book_info.ratings_count,
                description=book_info.description,
                book_identifier=book_info.get_primary_isbn13(),
                image_link=book_info.image_links.get("smallThumbnail", "NA")
            )

        except Exception as e:
            print(f"Error processing {metadata.title}: {str(e)}")
            return None
