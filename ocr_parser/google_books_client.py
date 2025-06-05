import aiohttp
import asyncio
from typing import Optional, Dict

from ocr_parser.interfaces import IBookApiClient
from ocr_parser.models import BookInfo, BookIdentifier


class GoogleBooksClient(IBookApiClient):

    def __init__(self, max_concurrent_requests: int = 5, retry_attempts: int = 3, retry_delay: int = 2):
        self.max_concurrent_requests = max_concurrent_requests
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay
        self.semaphore = asyncio.Semaphore(max_concurrent_requests)

    async def search_book(self, title: str, author: str) -> Optional[BookInfo]:
        async with self.semaphore:
            for attempt in range(self.retry_attempts):
                try:
                    query = f'intitle:"{title}"+inauthor:"{author}"'
                    url = f'https://www.googleapis.com/books/v1/volumes?q={query}'

                    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=False)) as session:
                        async with session.get(url) as response:
                            if response.status == 429:  # Too Many Requests
                                await asyncio.sleep(self.retry_delay * (attempt + 1))
                                continue

                            data = await response.json()
                            return self._process_api_response(data, title, author)

                except Exception as e:
                    if attempt < self.retry_attempts - 1:
                        await asyncio.sleep(self.retry_delay * (attempt + 1))
                        continue
                    print(f"Error searching for {title} by {author}: {str(e)}")
                    return None

            return None

    def _process_api_response(self, data: Dict, title: str, author: str) -> Optional[BookInfo]:
        if 'items' not in data:
            return None

        items = data['items']
        relevant_items = [item for item in items if self.__is_relevant(item, title, author)]

        if not relevant_items:
            return None

        sorted_items = sorted(
            relevant_items,
            key=lambda x: (
                self.get_rating_count(x),
                x.get("volumeInfo", {}).get("averageRating", 0)
            ),
            reverse=True
        )

        best_match = sorted_items[0]
        return self.__create_book_info(best_match)

    def __is_relevant(self, volume: Dict, title: str, author: str) -> bool:
        info = volume.get("volumeInfo", {})
        return (
                title.lower() in info.get("title", "").lower() and
                any(author.lower() in a.lower() for a in info.get("authors", []))
        )

    def get_rating_count(self, volume: Dict) -> int:
        return volume.get("volumeInfo", {}).get("ratingsCount", 0)

    def __create_book_info(self, volume: Dict) -> BookInfo:
        info = volume["volumeInfo"]
        identifiers_raw = info.get("industryIdentifiers", [])
        identifiers = [
            BookIdentifier(
                type=id_info.get("type", "UNKNOWN"),
                value=id_info.get("identifier", "NA")
            )
            for id_info in identifiers_raw
        ]

        return BookInfo(
            title=info.get("title", "NA"),
            author=", ".join(info.get("authors", [])),
            identifiers=identifiers,
            genres=info.get("categories", ["NA"]),
            average_rating=float(info.get("averageRating", 0)),
            ratings_count=int(info.get("ratingsCount", 0)),
            description=info.get("description", "NA"),
            image_links=info.get("imageLinks", {})
        )
