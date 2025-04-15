from langchain.chat_models import init_chat_model
from models.query_keywords import QueryKeywords
from services.paper_retriever import PaperRetriever


class ResearchAgent:
    """
    ResearchAgent class to handle the research process based on command-line arguments provided.
    """

    def __init__(self, args):
        """
        Initialize the research agent with parsed arguments.
        :param args: Parsed command-line arguments.
        """
        self.query = args.query
        self.start_date = args.start_date
        self.end_date = args.end_date
        self.paper_count = args.paper_count
        self.focus = args.focus
        self.llm_model = init_chat_model("claude-3-5-sonnet-latest", model_provider="anthropic", temperature=0)

    def research_pipeline(self):
        """
        The main research pipeline to perform literature review.
        """
        print("Executing the research pipeline...")
        print(f"Query: {self.query}")
        #keywords = self.get_query_keywords(self.query)
        keywords = ['person re-identification', 'neural networks', 'occlusion handling', 'multi-person tracking', 'deep learning', 'person ReID', 'occlusion robust', 'identity disambiguation', 'pedestrian tracking']
        keywords = ['person re-identification']
        print("Searching articles in Semantic Scholar database...")
        retriever = PaperRetriever()
        papers = retriever.retrieve_papers(
            keywords=keywords,
            start_date=self.start_date,
            end_date=self.end_date,
            focus=self.focus,
            max_papers=10
        )
        retriever.print_papers(papers)


    def get_query_keywords(self, query):
        structured_llm = self.llm_model.with_structured_output(QueryKeywords)
        result = structured_llm.invoke(query)
        return result.keywords

