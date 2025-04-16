import os
import chromadb
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

class VectorDb:
    def __init__(self, base_dir, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the VectorDb class.
        
        Args:
            base_dir (str): The base directory where the vector database will be stored
            model_name (str): The embedding model name to use
        """
        self.db_directory = os.path.join(base_dir, "vector_db")
        os.makedirs(self.db_directory, exist_ok=True)
        self.model_name = model_name
        
        # Initialize embeddings model
        self.embeddings = HuggingFaceEmbeddings(model_name=self.model_name)
        
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
            papers_added = self.prepare_documents(documents, existing_ids, ids, metadatas, papers, papers_added)
            
            # Add documents to the existing database
            if documents:
                vectordb.add_texts(
                    texts=documents,
                    metadatas=metadatas,
                    ids=ids
                )
                print(f"Added embeddings for {papers_added} new papers to the existing database")
            else:
                print("No new papers to add to the database")
                
        else:
            # Create a new database
            if not append and db_exists:
                print("Creating a new vector database (overwriting existing one)")
            
            # Prepare documents for the vector database
            documents = []
            metadatas = []
            ids = []
            existing_ids = set()
            papers_added = 0
            papers_added = self.prepare_documents(documents, existing_ids, ids, metadatas, papers, papers_added)
            
            # Create the Chroma vector store
            vectordb = Chroma.from_texts(
                texts=documents,
                embedding=self.embeddings,
                metadatas=metadatas,
                ids=ids,
                persist_directory=self.db_directory
            )
            print(f"Created a new database with embeddings for {len(documents)} papers")
        
        return vectordb

    def prepare_documents(self, documents, existing_ids, ids, metadatas, papers, papers_added):
        for paper in papers:
            if 'abstract' in paper and paper['abstract'] and paper.get('paperId'):
                paper_id = f"paper_{paper.get('paperId')}"

                # Skip if this paper is already in the database
                if paper_id in existing_ids:
                    continue

                documents.append(paper['abstract'])

                # Create metadata for each paper
                metadata = {
                    'title': paper.get('title', ''),
                    'authors': ', '.join([author.get('name', '') for author in paper.get('authors', [])]),
                    'year': paper.get('year', ''),
                    'url': paper.get('url', ''),
                    'paperId': paper.get('paperId', '')
                }
                metadatas.append(metadata)
                ids.append(paper_id)
                papers_added += 1
        return papers_added

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
            print("Vector database not found. Create it first by calling create_embeddings_and_store().")
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