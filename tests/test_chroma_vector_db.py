import unittest
from unittest.mock import Mock, patch, MagicMock
import os
import tempfile
from classes.vector_db.chroma_vector_db import ChromaVectorDb
from utils.error_handler import DatabaseError

class TestChromaVectorDb(unittest.TestCase):
    
    def setUp(self):
        # Create a temporary directory for test DB
        self.temp_dir = tempfile.TemporaryDirectory()
        # Mock HuggingFaceEmbeddings
        self.mock_embeddings = Mock()
        with patch('classes.vector_db.chroma_vector_db.HuggingFaceEmbeddings', return_value=self.mock_embeddings):
            self.vector_db = ChromaVectorDb(self.temp_dir.name, model_name="test-model")
    
    def tearDown(self):
        self.temp_dir.cleanup()
    
    @patch('classes.vector_db.chroma_vector_db.Chroma')
    @patch('classes.vector_db.chroma_vector_db.os.path.exists')
    @patch('classes.vector_db.chroma_vector_db.DocumentProcessor.prepare_documents')
    def test_create_embeddings_and_store_new_db(self, mock_prepare_docs, mock_exists, mock_chroma):
        # Arrange
        mock_exists.return_value = False
        mock_prepare_docs.return_value = 5  # 5 papers added
        mock_chroma_instance = MagicMock()
        mock_chroma.from_texts.return_value = mock_chroma_instance
        
        test_papers = [{"title": "Test Paper", "abstract": "Test abstract"}]
        mock_documents = ["doc1", "doc2"]
        mock_metadatas = [{"meta1": "data1"}, {"meta2": "data2"}]
        mock_ids = ["id1", "id2"]
        
        # Act
        result = self.vector_db.create_embeddings_and_store(test_papers, append=False)
        
        # Assert
        self.assertEqual(result, mock_chroma_instance)
        mock_chroma.from_texts.assert_called_once()
    
    @patch('classes.vector_db.chroma_vector_db.Chroma')
    @patch('classes.vector_db.chroma_vector_db.os.path.exists')
    def test_query_vector_database(self, mock_exists, mock_chroma):
        # Arrange
        mock_exists.return_value = True
        mock_chroma_instance = MagicMock()
        mock_chroma.return_value = mock_chroma_instance
        
        # Mock search results
        doc1 = MagicMock()
        doc1.page_content = "content1"
        doc1.metadata = {"title": "title1"}
        doc2 = MagicMock()
        doc2.page_content = "content2"
        doc2.metadata = {"title": "title2"}
        
        mock_chroma_instance.similarity_search_with_score.return_value = [
            (doc1, 0.9),
            (doc2, 0.8)
        ]
        
        # Act
        results = self.vector_db.query_vector_database("test query", n_results=2)
        
        # Assert
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]['content'], "content1")
        self.assertEqual(results[0]['metadata'], {"title": "title1"})
        self.assertEqual(results[0]['similarity_score'], 0.9)
        self.assertEqual(results[1]['content'], "content2")
        self.assertEqual(results[1]['similarity_score'], 0.8)
        
        mock_chroma.assert_called_once_with(
            persist_directory=self.vector_db.db_directory,
            embedding_function=self.mock_embeddings
        )
        mock_chroma_instance.similarity_search_with_score.assert_called_once_with("test query", k=2)
    
    @patch('classes.vector_db.chroma_vector_db.os.path.exists')
    def test_query_nonexistent_database(self, mock_exists):
        # Arrange
        mock_exists.return_value = False
        
        # Act
        results = self.vector_db.query_vector_database("test query")
        
        # Assert
        self.assertEqual(results, [])

if __name__ == '__main__':
    unittest.main()