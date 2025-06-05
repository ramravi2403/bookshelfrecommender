import asyncio

from ocr_parser.book_processing_factory import BookProcessingFactory


async def main():

    sample_ocr_texts = [
        "left hand darkness le guin",
        "dune messiah herbert",
        "crime punishment penguin"
    ]

    processing_service = BookProcessingFactory.create_processing_service(
        model_name="gemini-2.0-flash",
        batch_size=5
    )
    output_file = "processed_books_dataset.csv"
    await processing_service.process_ocr_texts_to_dataset(sample_ocr_texts, output_file)


if __name__ == "__main__":
    asyncio.run(main())