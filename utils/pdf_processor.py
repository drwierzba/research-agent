import os
import base64
from io import BytesIO
from typing import List, Dict, Any
import fitz  # PyMuPDF

from config.app_config import AppConfig
from utils.error_handler import handle_exceptions, ResearchAgentError


class PDFProcessor:
    def __init__(self, max_pages_per_pdf: int = AppConfig.MAX_PAGES_PER_PDF):
        """
        Initialize PDFProcessor.

        Args:
            max_pages_per_pdf (int): Maximum number of pages to process per PDF
        """
        self.max_pages_per_pdf = max_pages_per_pdf

    @handle_exceptions(error_type=ResearchAgentError, default_return=([], []))
    def process_pdf_documents(self, documents: List[Dict[str, Any]]) -> tuple:
        """
        Process PDF documents to extract metadata and convert pages to base64 images.

        Args:
            documents: List of document dictionaries from vector_db.query_vector_database

        Returns:
            tuple: (metadata_list, pdf_images) where metadata_list is a list of document metadata
                  and pdf_images is a list of base64-encoded images of PDF pages
        """
        metadata_list = []
        pdf_images = []

        for i, doc in enumerate(documents, 1):
            metadata = doc.get('metadata', {})
            pdf_path = metadata.get('local_file_path', '')

            if pdf_path == '' or not pdf_path or not os.path.exists(pdf_path):
                continue

            # Add document metadata to the list
            doc_info = {
                'document_number': i,
                'title': metadata.get('title', 'Unknown'),
                'authors': metadata.get('authors', 'Unknown'),
                'year': metadata.get('year', 'Unknown'),
                'similarity_score': doc.get('similarity_score', 0)
            }
            metadata_list.append(doc_info)

            # Convert PDF pages to images
            page_images = self._pdf_to_base64_images(pdf_path)
            pdf_images.extend(page_images)

        return metadata_list, pdf_images

    def _pdf_to_base64_images(self, pdf_path: str) -> List[str]:
        """
        Convert PDF pages to base64-encoded PNG images.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            List[str]: List of base64-encoded PNG images, one per page
        """
        base64_images = []

        # Open the PDF
        pdf_document = fitz.open(pdf_path)

        # Limit the number of pages to process
        page_count = min(len(pdf_document), self.max_pages_per_pdf)

        # Convert each page to a PNG image
        for page_num in range(page_count):
            page = pdf_document.load_page(page_num)

            # Set a reasonable resolution (higher values = better quality but larger files)
            zoom = 2.0
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat)

            # Convert pixmap to PNG image
            img_bytes = BytesIO(pix.pil_tobytes(format="PNG"))

            # Convert to base64
            base64_image = base64.b64encode(img_bytes.getvalue()).decode('utf-8')
            base64_images.append(base64_image)

        pdf_document.close()

        return base64_images

