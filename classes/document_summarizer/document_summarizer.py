from abc import ABC, abstractmethod


class DocumentSummarizer(ABC):
    @abstractmethod
    def create_summary(self, documents):
        pass
