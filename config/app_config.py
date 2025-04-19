# config/app_config.py
import os
from dataclasses import dataclass

@dataclass
class AppConfig:
    """Application configuration"""
    # Paths
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    PAPERS_DIR: str = os.path.join(BASE_DIR, "data/papers/raw")

    # API settings
    DEFAULT_PAPER_COUNT: int = 20

    # Model settings
    DEFAULT_EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    MAX_PAGES_PER_PDF: int = 20

    # Database
    VECTOR_DB_FOLDER: str = "vector_db"