from pydantic import BaseModel, Field
from typing import List


class QueryKeywords(BaseModel):
    query: str = Field(description="Query to be transformed into the list of keywords for Semantic Scholar API search")
    keywords: List[str] = Field(description="List of article/paper keywords for Semantic Scholar API search, maximum 3")
