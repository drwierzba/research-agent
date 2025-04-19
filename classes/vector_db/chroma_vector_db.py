import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from classes.vector_db.vector_database import VectorDatabase
from config.app_config import AppConfig
from utils.document_processor import DocumentProcessor
from utils.error_handler import handle_exceptions, DatabaseError
from utils.logger import Logger


class ChromaVectorDb(VectorDatabase):
    def __init__(self, base_dir, model_name=AppConfig.DEFAULT_EMBEDDING_MODEL):
        """
        Initialize the VectorDb class.
        
        Args:
            base_dir (str): The base directory where the vector database will be stored
            model_name (str): The embedding model name to use
        """
        self.logger = Logger.get_logger(self.__class__.__name__)
        self.db_directory = os.path.join(base_dir, AppConfig.VECTOR_DB_FOLDER)
        os.makedirs(self.db_directory, exist_ok=True)
        self.model_name = model_name
        
        # Initialize embeddings model
        self.embeddings = HuggingFaceEmbeddings(model_name=self.model_name)

    @handle_exceptions(error_type=DatabaseError)
    def create_embeddings_and_store(self, papers, append=True):
        """
        Create embeddings for paper abstracts and store them in a Chroma vector database.
        
        Args:
            papers (list): List of paper dictionaries containing abstracts
            append (bool): If True, append to existing database; if False, create new database
            
        Returns:
            Chroma: The Chroma vector database instance
        """
        # Check if the database already exists
        db_exists = os.path.exists(os.path.join(self.db_directory, "chroma.sqlite3"))
        
        if db_exists and append:
            # Load existing database
            vectordb = Chroma(persist_directory=self.db_directory, embedding_function=self.embeddings)
            
            # Get existing paper IDs to avoid duplicates
            existing_ids = set(vectordb._collection.get()["ids"])
            
            # Prepare documents for the vector database
            documents = []
            metadatas = []
            ids = []
            
            # Add papers that aren't already in the database
            papers_added = 0
            papers_added = DocumentProcessor.prepare_documents(documents, existing_ids, ids, metadatas, papers, papers_added)
            
            # Add documents to the existing database
            if documents:
                vectordb.add_texts(
                    texts=documents,
                    metadatas=metadatas,
                    ids=ids
                )
                Logger.info(self.logger, f"Added embeddings for {papers_added} new papers to the existing database")
            else:
                Logger.info(self.logger, "No new papers to add to the database")
                
        else:
            # Create a new database
            if not append and db_exists:
                Logger.info(self.logger, "Creating a new vector database (overwriting existing one)")
            
            # Prepare documents for the vector database
            documents = []
            metadatas = []
            ids = []
            existing_ids = set()
            papers_added = 0
            papers_added = DocumentProcessor.prepare_documents(documents, existing_ids, ids, metadatas, papers, papers_added)
            
            # Create the Chroma vector store
            vectordb = Chroma.from_texts(
                texts=documents,
                embedding=self.embeddings,
                metadatas=metadatas,
                ids=ids,
                persist_directory=self.db_directory
            )
            Logger.info(self.logger, f"Created a new database with embeddings for {len(documents)} papers")
        
        return vectordb

    @handle_exceptions(error_type=DatabaseError, default_return= [])
    def query_vector_database(self, query, n_results=5):
        """
        Query the vector database for papers similar to the query.
        
        Args:
            query (str): The query string
            n_results (int): Number of results to return
            
        Returns:
            list: List of papers similar to the query
        """
        # Check if the vector database exists
        if not os.path.exists(os.path.join(self.db_directory, "chroma.sqlite3")):
            Logger.info(self.logger, "Vector database not found. Create it first by calling create_embeddings_and_store().")
            return []
        
        # Load the vector database
        vectordb = Chroma(persist_directory=self.db_directory, embedding_function=self.embeddings)
        
        # Query the database
        results = vectordb.similarity_search_with_score(query, k=n_results)
        
        # Format and return the results
        formatted_results = []
        for doc, score in results:
            result = {
                'content': doc.page_content,
                'metadata': doc.metadata,
                'similarity_score': score
            }
            formatted_results.append(result)
            
        return formatted_results