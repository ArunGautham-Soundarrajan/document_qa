from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from typing import List


# TODO
def create_db(documents: List[Document]) -> Chroma:
    # Load the embedding model
    embeddings = HuggingFaceInstructEmbeddings(
        query_instruction="Represent the query for retrieval: "
    )
    db = Chroma.from_documents(documents, embeddings)
    return db
