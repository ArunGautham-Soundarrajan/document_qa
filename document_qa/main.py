from vector_search.vector_db import create_db
from pprint import pprint


def main():
    pdf_path = "document_qa/documents/general-accident-financial-statements.pdf"
    db = create_db(pdf_path=pdf_path, pages=(0, 3))

    query = "What are the contents"
    docs = db.similarity_search(query)
    pprint(docs[0])


if __name__ == "__main__":
    main()
