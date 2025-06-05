from typing import List

import pandas as pd

from ocr_parser.interfaces import IDatasetExporter
from ocr_parser.models import DatasetRecord


class CsvDatasetExporter(IDatasetExporter):

    def export(self, records: List[DatasetRecord], output_path: str) -> None:
        if not records:
            print("No records to export")
            return
        data = []
        for record in records:
            data.append({
                "Title": record.title,
                "Author": record.author,
                "Genres": record.genres,
                "Rating": record.rating,
                "RatingCount": record.rating_count,
                "Description": record.description,
                "BookIdentifier": record.book_identifier,
                "imageLink": record.image_link
            })

        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False)

        print(f"Dataset exported with {len(records)} entries to: {output_path}")
        print("\nFirst few entries:")
        print(df.head())