import pandas as pd
import numpy as np
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import re
from typing import List, Dict, Any, Optional, Tuple

class SemanticBookRecommender:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.sentence_model = SentenceTransformer(model_name)
        self.emotion_classifier = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            return_all_scores=True
        )
        self.genre_classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli"
        )
        self.books_df = None
        self.book_embeddings = None
        self.available_genres = set()
        self.available_emotions = set()

    def load_and_preprocess_data(self, filepath: str) -> None:
        self.books_df = pd.read_csv(filepath)
        self.books_df = self.books_df.dropna(subset=['Description'])
        self.books_df['Author'] = self.books_df['Author'].str.strip()
        self.books_df['Title'] = self.books_df['Title'].str.strip()
        self.books_df = self.books_df.drop_duplicates(subset=['Author', 'Title'])
        self.books_df = self.books_df.reset_index(drop=True)
        self.books_df['Emotion'] = self.books_df['Description'].apply(self.predict_emotion)
        self.books_df['Genres'] = self.books_df['Genres'].fillna('Unknown')
        self._create_embeddings()
        self._update_available_options()

    def _create_embeddings(self) -> None:
        texts = self.books_df.apply(
            lambda row: f"Title: {row['Title']} | Author: {row['Author']} | Description: {row['Description']}", axis=1
        )
        self.book_embeddings = self.sentence_model.encode(texts.tolist(), show_progress_bar=True)

    def predict_emotion(self, text: str) -> str:
        scores = self.emotion_classifier(text)[0]
        return max(scores, key=lambda x: x['score'])['label']

    def _update_available_options(self):
        all_genres = []
        for g in self.books_df['Genres']:
            all_genres.extend([x.strip() for x in str(g).split(',')])
        self.available_genres = set(all_genres)
        self.available_emotions = set(self.books_df['Emotion'].dropna().unique())

    def parse_query_filters(self, query: str) -> Tuple[str, Dict[str, Any]]:
        filters = {}
        clean_query = query.lower()

        rating_patterns = [
            (r'rating[s]?[\s]*[>]\s*(\d+\.?\d*)', 'min_rating'),
            (r'rating[s]?[\s]*[<]\s*(\d+\.?\d*)', 'max_rating'),
            (r'[>]\s*(\d+\.?\d*)\s*star[s]?', 'min_rating'),
            (r'[<]\s*(\d+\.?\d*)\s*star[s]?', 'max_rating')
        ]

        for pattern, key in rating_patterns:
            match = re.search(pattern, clean_query)
            if match:
                filters[key] = float(match.group(1))
                clean_query = re.sub(pattern, '', clean_query)

        author_match = re.search(r'by\s+([a-zA-Z\s\.]+)', clean_query)
        if author_match:
            filters['author'] = author_match.group(1).strip()
            clean_query = re.sub(r'by\s+[a-zA-Z\s\.]+', '', clean_query)

        return clean_query.strip(), filters

    def apply_filters(self, df: pd.DataFrame, filters: Dict[str, Any]) -> pd.DataFrame:
        result = df.copy()
        if 'min_rating' in filters:
            result = result[result['Rating'] >= filters['min_rating']]
        if 'max_rating' in filters:
            result = result[result['Rating'] <= filters['max_rating']]
        if 'author' in filters:
            result = result[result['Author'].str.lower().str.contains(filters['author'].lower(), na=False)]
        return result

    def find_similar_books(
        self,
        query: str,
        top_k: int = 5,
        genre_filter: Optional[str] = None,
        emotion_filter: Optional[str] = None
    ) -> pd.DataFrame:
        clean_query, query_filters = self.parse_query_filters(query)
        query_embedding = self.sentence_model.encode(clean_query)

        similarities = np.dot(self.book_embeddings, query_embedding) / (
            np.linalg.norm(self.book_embeddings, axis=1) * np.linalg.norm(query_embedding)
        )

        top_indices = np.argsort(similarities)[-top_k:][::-1]
        results = self.books_df.iloc[top_indices]

        if genre_filter and genre_filter != "All":
            results = results[results['Genres'].str.contains(genre_filter, case=False, na=False)]
        if emotion_filter and emotion_filter != "All":
            results = results[results['Emotion'] == emotion_filter]

        results = self.apply_filters(results, query_filters)

        return results.reset_index(drop=True)

    def get_available_genres(self) -> List[str]:
        return ["All"] + sorted(self.available_genres)

    def get_available_emotions(self) -> List[str]:
        return ["All"] + sorted(self.available_emotions)

    def get_rating_stats(self) -> Dict[str, float]:
        r = self.books_df['Rating']
        return {'min': float(r.min()), 'max': float(r.max()), 'mean': float(r.mean())}
