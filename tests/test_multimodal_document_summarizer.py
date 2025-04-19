import unittest
from unittest.mock import Mock, patch
from classes.document_summarizer.multimodal_document_summarizer import MultimodalDocumentSummarizer
from classes.model_adapter.model_adapter import ModelAdapter
from utils.pdf_processor import PDFProcessor

class TestMultimodalDocumentSummarizer(unittest.TestCase):
    
    def setUp(self):
        self.mock_model_adapter = Mock(spec=ModelAdapter)
        self.mock_pdf_processor = Mock(spec=PDFProcessor)
        self.summarizer = MultimodalDocumentSummarizer(
            focus="Test focus",
            model_adapter=self.mock_model_adapter,
            pdf_processor=self.mock_pdf_processor
        )
    
    def test_create_summary_no_documents(self):
        # Act
        result = self.summarizer.create_summary([])
        
        # Assert
        self.assertEqual(result, "No documents provided for summarization.")
        self.mock_pdf_processor.process_pdf_documents.assert_not_called()
        self.mock_model_adapter.invoke.assert_not_called()
    
    def test_create_summary_with_images_method(self):
        # Arrange
        documents = [{"id": "doc1"}, {"id": "doc2"}]
        metadata_list = [
            {"document_number": 1, "title": "Paper 1", "authors": "Author 1", "year": "2023", "similarity_score": 0.95},
            {"document_number": 2, "title": "Paper 2", "authors": "Author 2", "year": "2022", "similarity_score": 0.85}
        ]
        pdf_images = ["base64_1", "base64_2"]
        self.mock_pdf_processor.process_pdf_documents.return_value = (metadata_list, pdf_images)
        
        mock_model_with_images = Mock()
        self.mock_model_adapter.with_images.return_value = mock_model_with_images
        mock_model_with_images.invoke.return_value = "Generated summary"
        
        # Act
        result = self.summarizer.create_summary(documents)
        
        # Assert
        self.assertEqual(result, "Generated summary")
        self.mock_pdf_processor.process_pdf_documents.assert_called_once_with(documents)
        self.mock_model_adapter.with_images.assert_called_once_with(pdf_images)
        mock_model_with_images.invoke.assert_called_once()
        # Verify that prompt contains expected metadata
        prompt = mock_model_with_images.invoke.call_args[0][0]
        self.assertIn("Paper 1", prompt)
        self.assertIn("Paper 2", prompt)
        self.assertIn("Test focus", prompt)
    
    def test_create_summary_with_invoke_with_images_method(self):
        # Arrange
        documents = [{"id": "doc1"}]
        metadata_list = [
            {"document_number": 1, "title": "Paper 1", "authors": "Author 1", "year": "2023", "similarity_score": 0.95}
        ]
        pdf_images = ["base64_1"]
        self.mock_pdf_processor.process_pdf_documents.return_value = (metadata_list, pdf_images)
        
        # Remove with_images method to test alternate path
        delattr(self.mock_model_adapter, 'with_images')
        self.mock_model_adapter.invoke_with_images = Mock(return_value="Generated summary")
        
        # Act
        result = self.summarizer.create_summary(documents)
        
        # Assert
        self.assertEqual(result, "Generated summary")
        self.mock_pdf_processor.process_pdf_documents.assert_called_once_with(documents)
        # Verify appropriate message format was used
        expected_message = {
            "role": "user", 
            "content": [
                {"type": "text", "text": self.summarizer._create_summarization_prompt(metadata_list)}
            ]
        }
        # Add image attachments to expected message
        expected_message["content"].append({
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64,base64_1"}
        })
        self.mock_model_adapter.invoke_with_images.assert_called_once()

    def test_create_summary_no_multimodal_support(self):
        # Arrange
        documents = [{"id": "doc1"}]
        metadata_list = [{"document_number": 1, "title": "Paper 1", "authors": "Author 1", "year": "2023", "similarity_score": 0.95}]
        pdf_images = ["base64_1"]
        self.mock_pdf_processor.process_pdf_documents.return_value = (metadata_list, pdf_images)
        
        # Remove both multimodal methods
        delattr(self.mock_model_adapter, 'with_images')
        
        # Act
        result = self.summarizer.create_summary(documents)
        
        # Assert
        self.assertEqual(result, "Error: The provided LLM model does not support multimodal inputs with images.")

if __name__ == '__main__':
    unittest.main()