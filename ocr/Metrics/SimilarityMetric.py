from abc import ABC, abstractmethod


class SimilarityMetric(ABC):
    def __init__(self, metric_name:str):
        self.__metric_name = metric_name

    def name(self) -> str:
        return self.__metric_name

    @abstractmethod
    def similarity(self, a: str, b: str) -> float:
        pass
