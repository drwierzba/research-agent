from abc import ABC, abstractmethod

class VectorDatabase(ABC):
    """Interface for vector database implementations"""

    @abstractmethod
    def create_embeddings_and_store(self, documents, append=True):
        """Store document embeddings in the database"""
        pass

    @abstractmethod
    def query_vector_database(self, query, n_results=5):
        """Query the database for similar documents"""
        pass