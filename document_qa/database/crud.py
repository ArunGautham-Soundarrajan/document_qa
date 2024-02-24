import uuid

from database.models import Documents
from database.schemas import DocumentBase, Item
from sqlalchemy.orm import Session


def get_document(db: Session, document_id: str):
    """Get Document details with document id

    :param Session db: Database session
    :param str document_id: ID of the document
    :return _type_: Document information
    """
    return db.query(Documents).filter(Item.id == document_id).first()


def get_all_documents(db: Session, skip: int = 0, limit: int = 100):
    """Get all the documents in the db

    :param Session db: Database session
    :param int skip: Number of documents to skip in search, defaults to 0
    :param int limit: Number of top documents to retrieve, defaults to 100
    :return _type_: List of documents
    """
    return db.query(Documents).offset(skip).limit(limit).all()


def create_document_item(db: Session, item: DocumentBase):
    """Create a document item

    :param Session db: Database session
    :param DocumentBase item: Pydantic document information class
    :return _type_: Index of the created document
    """
    db_item = Documents(**item.model_dump(), id=str(uuid.uuid4()))
    db.add(db_item)
    db.commit()
    db.refresh(db_item)
    return db_item
