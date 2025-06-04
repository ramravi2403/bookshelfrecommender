from Metrics.SimilarityMetric import SimilarityMetric


class Jaccard(SimilarityMetric):
    def __init__(self):
        super().__init__('Jaccard')

    def similarity(self, a: str, b: str) -> float:
        set_a = set(a)
        set_b = set(b)
        if not set_a and not set_b:
            return 1.0
        return len(set_a & set_b) / len(set_a | set_b)
