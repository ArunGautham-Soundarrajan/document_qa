from typing import Dict, List

from pydantic import BaseModel


class SearchResult(BaseModel):
    generated_answer: str
    sources: List[str]
