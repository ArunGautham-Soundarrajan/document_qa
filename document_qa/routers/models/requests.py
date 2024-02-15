from typing import List

from pydantic import BaseModel


class SearchItem(BaseModel):
    doc_id: str | List[str]
    question: str
