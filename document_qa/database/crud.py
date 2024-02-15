import uuid

from database.models import Documents
from database.schemas import DocumentBase, Item
from sqlalchemy.orm import Session


def get_document(db: Session, document_id: str):
    return db.query(Documents).filter(Item.id == document_id).first()


def get_all_documents(db: Session, skip: int = 0, limit: int = 100):
    return db.query(Documents).offset(skip).limit(limit).all()


def create_document_item(db: Session, item: DocumentBase):
    db_item = Documents(**item.model_dump(), id=str(uuid.uuid4()))
    db.add(db_item)
    db.commit()
    db.refresh(db_item)
    return db_item
