import os
from models.query_keywords import QueryKeywords
from classes.document_summarizer.multimodal_document_summarizer import MultimodalDocumentSummarizer
from services.paper_retriever import PaperRetriever
from classes.vector_db.chroma_vector_db import ChromaVectorDb
from utils.logger import Logger


class ResearchAgent:
    """
    ResearchAgent class to handle the research process based on command-line arguments provided.
    """

    def __init__(self, args, model_adapter,
                 paper_retriever=None,
                 vector_db=None,
                 document_summarizer=None):
        """
        Initialize the research agent with parsed arguments.
        :param args: Parsed command-line arguments.
        """
        self.query = args.query
        self.start_date = args.start_date
        self.end_date = args.end_date
        self.paper_count = args.paper_count
        self.focus = args.focus
        self.model_adapter = model_adapter
        self.logger = Logger.get_logger(self.__class__.__name__)

        # Use provided components or create defaults
        self.paper_retriever = paper_retriever or PaperRetriever()
        self.vector_db = vector_db or ChromaVectorDb(os.path.dirname(self.paper_retriever.DOWNLOAD_DIR))
        self.document_summarizer = document_summarizer or MultimodalDocumentSummarizer(self.focus, self.model_adapter)

    def research_pipeline(self):
        """
        The main research pipeline to perform literature review.
        """
        Logger.info(self.logger,"Executing the research pipeline...")
        Logger.info(self.logger,f"Query: {self.query}")
        keywords = self.get_query_keywords(self.query)
        Logger.info(self.logger,f"Searching articles in Semantic Scholar database (keywords: {keywords})...")
        papers = self.paper_retriever.retrieve_papers(
            keywords=keywords,
            start_date=self.start_date,
            end_date=self.end_date,
            max_papers=10
        )
        # Store the papers in the vector database
        self.vector_db.create_embeddings_and_store(papers, append=True)

        # Query the vector database
        Logger.info(self.logger,"\nQuerying vector database for similar papers...")
        search_results = self.vector_db.query_vector_database(self.query, n_results=2)

        Logger.info(self.logger,"\nProducing papers summary...")
        summary = self.document_summarizer.create_summary(search_results)
        Logger.info(self.logger,f"\nSummary:\n{summary}")

    def get_query_keywords(self, query):
        result = self.model_adapter.with_structured_output(QueryKeywords,query)
        return result.keywords

