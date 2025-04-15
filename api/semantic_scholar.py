import requests
from typing import Dict, List, Optional, Tuple, Union
import urllib.parse
import time
import random
import logging

class SemanticScholarClient:
    """Client for interacting with the Semantic Scholar API."""

    BASE_URL = "https://api.semanticscholar.org/graph/v1"

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Semantic Scholar API client.

        Args:
            api_key (str, optional): API key for authenticated requests.
                                    Higher rate limits with an API key.
        """
        self.api_key = api_key
        self.headers = {}
        if api_key:
            self.headers["x-api-key"] = api_key

    def search_papers(
            self,
            query: str,
            year: Optional[Dict] = None,
            fields_of_study: Optional[List[str]] = None,
            limit: int = 10,
            offset: int = 0,
            fields: Optional[List[str]] = None,
            max_retries: int = 10,
            initial_backoff: float = 1.0,
            backoff_factor: float = 2.0,
            jitter: float = 0.1

    ) -> Dict:
        """
        Search for papers using keywords, date range, and fields of study.

        Args:
            query (str): Search query keywords
            year (tuple, optional): Tuple of (start_date, end_date) in format 'YYYY-MM-DD'
            fields_of_study (list, optional): List of fields of study to filter by
            limit (int, optional): Maximum number of results to return (default: 10)
            offset (int, optional): Index of the first result to return (default: 0)
            fields (list, optional): Fields to include in the response
            max_retries (int, optional): Maximum number of retry attempts for rate limit errors
            initial_backoff (float, optional): Initial backoff time in seconds
            backoff_factor (float, optional): Multiplicative factor for backoff after each retry
            jitter (float, optional): Random jitter factor to add to backoff times


        Returns:
            dict: JSON response from the API

        Raises:
            requests.HTTPError: If the API request fails
        """
        endpoint = f"{self.BASE_URL}/paper/search"

        # Prepare query parameters
        params = {
            "query": query,
            "limit": limit,
            "offset": offset,
            "openAccessPdf": ""
        }

        # Add year range if provided
        if year:
            start_year = year.get('start_year')
            end_year = year.get('end_year')

            if start_year:
                params["year"] = f">={start_year}"
            if end_year:
                if "year" in params:
                    params["year"] += f",<={end_year}"
                else:
                    params["year"] = f"<={end_year}"

        # Add fields of study if provided
        if fields_of_study and len(fields_of_study) > 0:
            params["fieldsOfStudy"] = ",".join(fields_of_study)

        # Add response fields if provided
        if fields and len(fields) > 0:
            params["fields"] = ",".join(fields)

        # Make the request with retry logic
        retry_count = 0
        backoff_time = initial_backoff

        while True:
            try:
                response = requests.get(endpoint, params=params, headers=self.headers)
                response.raise_for_status()
                return response.json()

            except requests.exceptions.HTTPError as e:
                # Check if we got a rate limit error (429)
                if e.response.status_code == 429 and retry_count < max_retries:
                    retry_count += 1

                    # Calculate sleep time with jitter
                    sleep_time = backoff_time * (1 + random.uniform(-jitter, jitter))

                    # Log the retry attempt
                    logging.warning(
                        f"Rate limit exceeded (429). Retrying in {sleep_time:.2f} seconds. "
                        f"Attempt {retry_count}/{max_retries}"
                    )

                    # Sleep before retrying
                    time.sleep(sleep_time)

                    # Increase backoff for next attempt
                    backoff_time *= backoff_factor

                else:
                    # Either it's not a 429 error or we've exceeded retries
                    raise


