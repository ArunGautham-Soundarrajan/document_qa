from vector_search.milvus_client import Milvus
from config import settings
from database.database import SessionLocal


async def get_db():
    """Create Milvus database session

    :yield _type_: database session
    """
    COLLECTION_NAME = settings.milvus.collection_name
    URI = settings.milvus.uri
    TOKEN = settings.milvus.token
    db = Milvus(uri=URI, token=TOKEN, collection_name=COLLECTION_NAME)
    try:
        yield db
    finally:
        db.close_connection()


def get_sql_db():
    """Create SQL lite database session

    :yield _type_: database session
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
