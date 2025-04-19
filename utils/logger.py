# utils/logger.py
import logging

class Logger:
    """Wrapper class for logging functionality"""

    @staticmethod
    def get_logger(name):
        """Get a logger with the specified name"""
        return logging.getLogger(name)

    @staticmethod
    def info(logger, message):
        """Log an info message"""
        logger.info(message)

    @staticmethod
    def warning(logger, message):
        """Log a warning message"""
        logger.warning(message)

    @staticmethod
    def error(logger, message, exc_info=False):
        """Log an error message"""
        logger.error(message, exc_info=exc_info)

    @staticmethod
    def debug(logger, message):
        """Log a debug message"""
        logger.debug(message)

    @staticmethod
    def critical(logger, message, exc_info=True):
        """Log a critical message"""
        logger.critical(message, exc_info=exc_info)