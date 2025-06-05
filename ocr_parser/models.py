from dataclasses import dataclass
from typing import List, Dict, Optional


@dataclass
class BookMetadata:
    title: str
    author: str
    isbn13: str

    def is_valid(self) -> bool:
        invalid_values = ['NA', 'None', 'Unknown', '']
        return (self.title not in invalid_values and
                self.author not in invalid_values)


@dataclass
class BookIdentifier:
    type: str
    value: str


@dataclass
class BookInfo:
    title: str
    author: str
    identifiers: List[BookIdentifier]
    genres: List[str]
    average_rating: float
    ratings_count: int
    description: str
    image_links: Dict[str, str]

    def get_primary_isbn13(self) -> str:
        if not self.identifiers:
            return "0"

        # First try to find ISBN-13
        for identifier in self.identifiers:
            if identifier.type.startswith('ISBN_13'):
                return identifier.value

        return self.identifiers[0].value if self.identifiers else "0"


@dataclass
class DatasetRecord:
    title: str
    author: str
    genres: str
    rating: float
    rating_count: int
    description: str
    book_identifier: str
    image_link: str
