import unittest
from unittest.mock import Mock, patch
from langchain import ResearchAgent
from models.query_keywords import QueryKeywords
from classes.document_summarizer.multimodal_document_summarizer import MultimodalDocumentSummarizer
from services.paper_retriever import PaperRetriever
from classes.vector_db.chroma_vector_db import ChromaVectorDb

class TestResearchAgent(unittest.TestCase):
    
    def setUp(self):
        # Create mocks for dependencies
        self.mock_model_adapter = Mock()
        self.mock_paper_retriever = Mock(spec=PaperRetriever)
        self.mock_vector_db = Mock(spec=ChromaVectorDb)
        self.mock_document_summarizer = Mock(spec=MultimodalDocumentSummarizer)
        
        # Setup the mock paper retriever
        self.mock_paper_retriever.DOWNLOAD_DIR = "/test/papers"
        
        # Create argument object with test values
        self.args = Mock()
        self.args.query = "Machine Learning"
        self.args.start_date = "2022-01-01"
        self.args.end_date = "2023-01-01"
        self.args.paper_count = 10
        self.args.focus = "Focus on applications in healthcare"
        
        # Create the ResearchAgent instance with mocked dependencies
        self.agent = ResearchAgent(
            args=self.args,
            model_adapter=self.mock_model_adapter,
            paper_retriever=self.mock_paper_retriever,
            vector_db=self.mock_vector_db,
            document_summarizer=self.mock_document_summarizer
        )
    
    def test_get_query_keywords(self):
        # Arrange
        expected_keywords = ["machine", "learning", "neural networks"]
        self.mock_model_adapter.with_structured_output.return_value = QueryKeywords(keywords=expected_keywords)
        
        # Act
        result = self.agent.get_query_keywords(self.args.query)
        
        # Assert
        self.assertEqual(result, expected_keywords)
        self.mock_model_adapter.with_structured_output.assert_called_once_with(
            QueryKeywords, 
            self.args.query
        )
    
    def test_research_pipeline(self):
        # Arrange
        keywords = ["machine", "learning"]
        test_papers = [{"title": "Test Paper", "abstract": "Abstract"}]
        test_search_results = [{"content": "Test content", "metadata": {"title": "Test"}}]
        summary = "This is a research summary."
        
        # Setup mock returns
        self.mock_model_adapter.with_structured_output.return_value = QueryKeywords(keywords=keywords)
        self.mock_paper_retriever.retrieve_papers.return_value = test_papers
        self.mock_vector_db.query_vector_database.return_value = test_search_results
        self.mock_document_summarizer.create_summary.return_value = summary
        
        # Act
        self.agent.research_pipeline()
        
        # Assert
        self.mock_model_adapter.with_structured_output.assert_called_once()
        self.mock_paper_retriever.retrieve_papers.assert_called_once_with(
            keywords=keywords,
            start_date=self.args.start_date,
            end_date=self.args.end_date,
            max_papers=10
        )
        self.mock_vector_db.create_embeddings_and_store.assert_called_once_with(
            test_papers,
            append=True
        )
        self.mock_vector_db.query_vector_database.assert_called_once_with(
            self.args.query,
            n_results=2
        )
        self.mock_document_summarizer.create_summary.assert_called_once_with(test_search_results)

if __name__ == '__main__':
    unittest.main()