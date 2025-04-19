import unittest
from unittest.mock import Mock, patch
from classes.model_adapter.claude_model_adapter import ClaudeModelAdapter
from models.query_keywords import QueryKeywords

class TestClaudeModelAdapter(unittest.TestCase):

    def setUp(self):
        self.mock_model = Mock()
        with patch('classes.model_adapter.claude_model_adapter.init_chat_model', return_value=self.mock_model):
            self.adapter = ClaudeModelAdapter()

    def test_invoke(self):
        # Arrange
        self.mock_model.invoke.return_value = "Test response"

        # Act
        result = self.adapter.invoke("Test prompt")

        # Assert
        self.assertEqual(result, "Test response")
        self.mock_model.invoke.assert_called_once_with("Test prompt")

    def test_invoke_with_images(self):
        # Arrange
        self.mock_model.invoke.return_value = "Test response with images"
        prompt = "Test prompt"
        images = ["base64_image_data1", "base64_image_data2"]
        expected_message = {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,base64_image_data1"}},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,base64_image_data2"}}
            ]
        }

        # Act
        result = self.adapter.invoke_with_images(prompt, images)

        # Assert
        self.assertEqual(result, "Test response with images")
        self.mock_model.invoke.assert_called_once_with([expected_message])

    def test_with_structured_output(self):
        # Arrange
        mock_structured_model = Mock()
        self.mock_model.with_structured_output.return_value = mock_structured_model
        mock_structured_model.invoke.return_value = QueryKeywords(keywords=["AI", "ML"])

        # Act
        result = self.adapter.with_structured_output(QueryKeywords, "Find keywords")

        # Assert
        self.assertEqual(result.keywords, ["AI", "ML"])
        self.mock_model.with_structured_output.assert_called_once_with(QueryKeywords)
        mock_structured_model.invoke.assert_called_once_with("Find keywords")

if __name__ == '__main__':
    unittest.main()