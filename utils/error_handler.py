import logging
from functools import wraps
from utils.logger import Logger

class ResearchAgentError(Exception):
    """Base exception for Research Agent errors"""
    pass

class APIError(ResearchAgentError):
    """Exception raised for API-related errors"""
    pass

class DatabaseError(ResearchAgentError):
    """Exception raised for database-related errors"""
    pass

def handle_exceptions(error_type=ResearchAgentError, default_return=None):
    """Decorator to handle exceptions in a consistent way"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get logger for the class if this is a method (args[0] is self)
            if args and hasattr(args[0], 'logger'):
                logger = args[0].logger
            else:
                # Fall back to module logger if not a class method
                logger = logging.getLogger(func.__module__)
                
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Get class name if method in a class
                class_name = args[0].__class__.__name__ if args else ""
                context = f"{class_name}.{func.__name__}" if class_name else func.__name__
                Logger.error(logger, f"Error in {context}: {str(e)}", exc_info=True)
                
                if default_return is not None:
                    return default_return
                raise error_type(f"Error in {context}: {str(e)}") from e
                
        return wrapper
    return decorator