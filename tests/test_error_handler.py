import unittest
from unittest.mock import Mock, patch
import logging
from utils.error_handler import handle_exceptions, ResearchAgentError, APIError, DatabaseError
from utils.logger import Logger

class TestErrorHandler(unittest.TestCase):
    
    def setUp(self):
        # Create a mock logger
        self.mock_logger = Mock()
        self.mock_logger.error = Mock()
        
        # Create a test class with a logger
        class TestClass:
            def __init__(self):
                self.logger = self.mock_logger
                
            @handle_exceptions(error_type=ResearchAgentError)
            def method_without_default(self):
                raise ValueError("Test error")
                
            @handle_exceptions(error_type=APIError, default_return="Default value")
            def method_with_default(self):
                raise ValueError("Test error")
                
        self.test_obj = TestClass()
        self.test_obj.logger = self.mock_logger
    
    def test_handle_exceptions_reraises_error(self):
        # Act/Assert
        with self.assertRaises(ResearchAgentError) as context:
            self.test_obj.method_without_default()
        
        # Verify the error message
        self.assertIn("Test error", str(context.exception))
        # Verify logger was called
        self.mock_logger.error.assert_called_once()
    
    def test_handle_exceptions_returns_default(self):
        # Act
        result = self.test_obj.method_with_default()
        
        # Assert
        self.assertEqual(result, "Default value")
        # Verify logger was called
        self.mock_logger.error.assert_called_once()
    
    @patch('utils.error_handler.logging.getLogger')
    def test_handle_exceptions_function(self, mock_get_logger):
        # Arrange
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        
        @handle_exceptions(error_type=DatabaseError, default_return=None)
        def test_function():
            raise ValueError("Function error")
        
        # Act/Assert
        with self.assertRaises(DatabaseError) as context:
            test_function()
        
        # Verify the error message
        self.assertIn("Function error", str(context.exception))
        # Verify logger was called
        mock_logger.error.assert_called_once()

if __name__ == '__main__':
    unittest.main()