import os
import sys
from typing import List, Dict, Optional, Any
from datetime import datetime

import requests

from api.semantic_scholar import SemanticScholarClient
import urllib.parse


class PaperRetriever:
    """
    A class to retrieve research papers based on keywords, date range, and focus area.
    This class uses the SemanticScholarClient to search for papers and retrieve their details.
    """
    DOWNLOAD_DIR = "data/papers/raw"

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the PaperRetriever class.

        Args:
            api_key (Optional[str]): The API key for Semantic Scholar.
                                    If not provided, will use default from SemanticScholarClient.
        """
        self.client = SemanticScholarClient(api_key=api_key)

    def retrieve_papers(self,
                        keywords: List[str],
                        start_date: Optional[str] = None,
                        end_date: Optional[str] = None,
                        focus: str = 'all',
                        max_papers: int = 20) -> List[Dict[str, Any]]:
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

        try:
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
                self.download_paper(result)

            return paper_details

        except Exception as e:
            print(f"Error during paper retrieval: {e}", file=sys.stderr)
            return []

    def download_paper(self, result):
        if 'openAccessPdf' in result and 'url' in result['openAccessPdf']:
            pdf_url = result['openAccessPdf']['url']
            paper_title = result.get('title', 'paper').replace("/", "_").replace(":","")  # Avoid invalid file characters
            file_name = f"{paper_title}.pdf"
            file_path = os.path.join(self.DOWNLOAD_DIR, file_name)

            try:
                # Download the PDF
                response = requests.get(pdf_url, stream=True)
                response.raise_for_status()

                # Write PDF to file
                with open(file_path, 'wb') as pdf_file:
                    for chunk in response.iter_content(chunk_size=1024):
                        pdf_file.write(chunk)

                print(f"Downloaded: {file_name}")
            except Exception as e:
                print(f"Error downloading {file_name} from {pdf_url}: {e}", file=sys.stderr)

    def print_papers(self, paper_details: List[Dict[str, Any]]) -> None:
        """
        Print details of retrieved papers to the console.

        Args:
            paper_details (List[Dict[str, Any]]): List of paper details to print
        """
        if not paper_details:
            print("No papers found matching the criteria.")
            return

        print(f"\n=== Found {len(paper_details)} papers ===\n")

        for i, paper in enumerate(paper_details, 1):
            print(f"Paper {i}:")
            print(f"Title: {paper.get('title', 'N/A')}")

            # Format authors
            authors = paper.get('authors', [])
            if authors:
                author_names = [author.get('name', 'Unknown') for author in authors]
                print(f"Authors: {', '.join(author_names)}")
            else:
                print("Authors: N/A")

            print(f"Year: {paper.get('year', 'N/A')}")
            print(f"Venue: {paper.get('venue', 'N/A')}")

            # Print abstract if available
            abstract = paper.get('abstract', None)
            if abstract:
                print(f"Abstract: {abstract[:200]}..." if len(abstract) > 200 else f"Abstract: {abstract}")

            # Citation information
            print(f"Citations: {paper.get('citationCount', 'N/A')}")

            # URL
            print(f"URL: {paper.get('url', 'N/A')}")

            print("-" * 50)
