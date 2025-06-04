from difflib import SequenceMatcher
from Metrics.SimilarityMetric import SimilarityMetric


class SequenceMatcherMetric(SimilarityMetric):
    def __init__(self):
        super().__init__('SequenceMatcher')

    def similarity(self, a: str, b: str) -> float:
        return SequenceMatcher(None, a, b).ratio()
