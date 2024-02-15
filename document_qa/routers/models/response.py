from typing import Dict

from pydantic import BaseModel


class SearchResult(BaseModel):
    generated_answer: str
    sources: Dict
