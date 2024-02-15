from pydantic import BaseModel


class DocumentBase(BaseModel):
    file_name: str
    processed: bool = False


class Item(DocumentBase):
    id: str
    file_name: str
    processed: bool

    class Config:
        orm_mode = True
