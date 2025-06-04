from Metrics.SimilarityMetric import SimilarityMetric


class Levenshtein(SimilarityMetric):
    def __init__(self):
        super().__init__('Levenshtein')

    def similarity(self, a: str, b: str) -> float:
        distance = Levenshtein.__levenshtein_distance(a, b)
        max_len = max(len(a), len(b))
        if max_len == 0:
            return 1.0
        return 1.0 - distance / max_len

    @staticmethod
    def __levenshtein_distance(a: str, b: str) -> int:
        m, n = len(a), len(b)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                cost = 0 if a[i - 1] == b[j - 1] else 1
                dp[i][j] = min(
                    dp[i - 1][j] + 1,  # deletion
                    dp[i][j - 1] + 1,  # insertion
                    dp[i - 1][j - 1] + cost  # substitution
                )
        return dp[m][n]
