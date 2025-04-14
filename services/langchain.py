from langchain.chat_models import init_chat_model
from models.query_keywords import QueryKeywords


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
        print(self.get_query_keywords(self.query))


    def get_query_keywords(self, query):
        structured_llm = self.llm_model.with_structured_output(QueryKeywords)
        result = structured_llm.invoke(query)
        return result.keywords

