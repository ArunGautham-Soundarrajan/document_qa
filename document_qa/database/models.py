from sqlalchemy import Boolean, Column, ForeignKey, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()


class Documents(Base):
    __tablename__ = "documents"

    id = Column(String, primary_key=True)
    file_name = Column(String)
    processed = Column(Boolean, default=False)
