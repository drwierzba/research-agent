from typing import List, Dict, Any, Optional

from classes.document_summarizer.document_summarizer import DocumentSummarizer
from classes.model_adapter.model_adapter import ModelAdapter
from config.app_config import AppConfig
from utils.pdf_processor import PDFProcessor
from utils.error_handler import handle_exceptions, ResearchAgentError


class MultimodalDocumentSummarizer(DocumentSummarizer):
    """
    Creates summaries of PDF documents retrieved from a vector database using 
    a multimodal LLM by attaching the PDF pages as images.
    """

    def __init__(self, focus: str, model_adapter: ModelAdapter,
                 pdf_processor: PDFProcessor = None,
                 max_pages_per_pdf: int = AppConfig.MAX_PAGES_PER_PDF):
        """
        Initialize the MultimodalDocumentSummarizer class.
        
        Args:
            max_pages_per_pdf: Maximum number of pages to process per PDF to avoid token limits
        """
        self.max_pages_per_pdf = max_pages_per_pdf
        self.focus = focus
        self.model_adapter = model_adapter
        self.pdf_processor = pdf_processor or PDFProcessor(max_pages_per_pdf=self.max_pages_per_pdf)

    @handle_exceptions(error_type=ResearchAgentError, default_return="Error generating summary")
    def create_summary(self, documents: List[Dict[str, Any]]) -> str:
        """
        Generate a summary of a list of documents using the provided multimodal LLM model.
        Instead of extracting text, this method renders PDF pages as images and attaches them
        to the prompt.

        Args:
            documents: List of document dictionaries from vector_db.query_vector_database
            llm_model: Multimodal LangChain compatible LLM model

        Returns:
            str: A summary of the provided documents
        """
        if not documents:
            return "No documents provided for summarization."

        # Extract document metadata and create PDF image attachments
        metadata_list, pdf_images = self.pdf_processor.process_pdf_documents(documents)
        
        # Create the prompt with document metadata
        prompt = self._create_summarization_prompt(metadata_list)
        
        # For multimodal LLMs, we need a special invocation with image attachments
        if hasattr(self.model_adapter, 'with_images'):
            # This is for models that support the with_images method
            response = self.model_adapter.with_images(pdf_images).invoke(prompt)
        elif hasattr(self.model_adapter, 'invoke_with_images'):
            # Alternative approach for some models
            message = {"role": "user", "content": [{"type": "text", "text": prompt}]}

            # Add image attachments to message
            for i, image in enumerate(pdf_images):
                message["content"].append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image}"}
                })

            response = self.model_adapter.invoke([message])
        else:
            # Fallback for basic models without multimodal support
            return "Error: The provided LLM model does not support multimodal inputs with images."

        return response

    def _create_summarization_prompt(self, metadata_list: List[Dict[str, Any]]) -> str:
        """
        Create a prompt for the LLM to generate a summary.
        
        Args:
            metadata_list: List of document metadata
            
        Returns:
            str: The complete prompt for summarization
        """
        # Create a document overview section
        document_overview = "\n".join([
            f"Document {meta['document_number']}:\n"
            f"  Title: {meta['title']}\n"
            f"  Authors: {meta['authors']}\n"
            f"  Year: {meta['year']}\n"
            f"  Relevance Score: {meta['similarity_score']}"
            for meta in metadata_list
        ])
        
        prompt = f"""
I'm attaching {len(metadata_list)} research papers as images for you to analyze. 
Here's information about the documents I'm providing:

{document_overview}

Please examine the attached PDF pages and create a comprehensive summary that:

1. Identifies the main research themes and questions across these papers
2. Highlights key methodologies used in the research
3. Synthesizes the main findings and conclusions
4. Notes any contradictions or differences in findings between the papers
5. Suggests potential areas for further research based on these papers
6. Indicates whether there is a Github repo related to the research paper or not

Your goal is to provide a cohesive summary that integrates information from all papers,
not to summarize each one separately. Focus on the most significant information.

{self.focus}
"""
        
        return prompt