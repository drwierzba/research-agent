import os
import sys
import requests

from utils.error_handler import handle_exceptions, APIError
from utils.logger import Logger
from config.app_config import AppConfig


class PaperDownloader:
    """Handles downloading and storing PDF files"""
    
    def __init__(self, download_dir=AppConfig.PAPERS_DIR):
        self.logger = Logger.get_logger(self.__class__.__name__)
        self.download_dir = download_dir
        os.makedirs(self.download_dir, exist_ok=True)

    @handle_exceptions(error_type=APIError)
    def download_paper(self, paper_metadata):
        if 'openAccessPdf' in paper_metadata and 'url' in paper_metadata['openAccessPdf']:
            pdf_url = paper_metadata['openAccessPdf']['url']
            paper_title = paper_metadata.get('title', 'paper').replace("/", "_").replace(":", "")  # Avoid invalid file characters
            file_name = f"{paper_title}.pdf"
            file_path = os.path.join(self.download_dir, file_name)
            paper_metadata['local_file_path'] = ""

            # Download the PDF
            response = requests.get(pdf_url, stream=True)
            response.raise_for_status()

            # Write PDF to file
            with open(file_path, 'wb') as pdf_file:
                for chunk in response.iter_content(chunk_size=1024):
                    pdf_file.write(chunk)
            paper_metadata['local_file_path'] = file_path
            Logger.info(self.logger, f"Downloaded: {file_name}")