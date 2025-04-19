import os
import sys
from typing import List, Dict, Optional, Any
from datetime import datetime
from api.semantic_scholar import SemanticScholarClient
import urllib.parse

from config.app_config import AppConfig
from utils.error_handler import handle_exceptions, APIError
from utils.paper_downloader import PaperDownloader


class PaperRetriever:
    """
    A class to retrieve research papers based on keywords, date range, and focus area.
    This class uses the SemanticScholarClient to search for papers and retrieve their details.
    """
    DOWNLOAD_DIR = AppConfig.PAPERS_DIR

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the PaperRetriever class.

        Args:
            api_key (Optional[str]): The API key for Semantic Scholar.
                                    If not provided, will use default from SemanticScholarClient.
        """
        self.client = SemanticScholarClient(api_key=api_key)
        self.downloader = PaperDownloader(download_dir=self.DOWNLOAD_DIR)

    @handle_exceptions(error_type=APIError, default_return=[])
    def retrieve_papers(self,
                        keywords: List[str],
                        start_date: Optional[str] = None,
                        end_date: Optional[str] = None,
                        max_papers: int = AppConfig.DEFAULT_PAPER_COUNT) -> List[Dict[str, Any]]:
        """
        Retrieve papers based on keywords, date range and focus.

        Args:
            keywords (List[str]): List of keywords to search for
            start_date (Optional[str]): Start date in format 'YYYY-MM-DD'
            end_date (Optional[str]): End date in format 'YYYY-MM-DD'
            focus (str): Focus area ('methodology', 'datasets', 'performance', 'all')
            max_papers (int): Maximum number of papers to retrieve

        Returns:
            List[Dict[str, Any]]: List of paper details
        """
        # Prepare query string from keywords
        encoded_keywords = [urllib.parse.quote(keyword) for keyword in keywords]
        query = "+".join(encoded_keywords)  # Use "+" to represent spaces in URL

        # Convert date strings to year ranges if provided
        year_filter = {}
        if start_date:
            year_filter["start_year"] = datetime.strptime(start_date, '%Y-%m-%d').year
        if end_date:
            year_filter["end_year"] = datetime.strptime(end_date, '%Y-%m-%d').year

        # Determine which fields to retrieve
        fields = ["title", "abstract", "year", "authors", "url", "paperId", "openAccessPdf"]
        # Search for papers
        search_results = self.client.search_papers(
            query=query,
            year=year_filter if year_filter else None,
            limit=max_papers,
            fields=fields
        )

        # Get detailed information for each paper
        paper_details = []
        # Directory to save downloaded files
        os.makedirs(self.DOWNLOAD_DIR, exist_ok=True)
        for result in search_results.get('data', []):
            paper_details.append(result)
            self.downloader.download_paper(result)

        return paper_details



