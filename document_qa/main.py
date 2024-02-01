# from vector_search.vector_db import create_db
from pprint import pprint

from text_extracter.pdf_miners import extract_text
from vector_search.text_splitter import TextSplitter
from vector_search.vector_db import create_db


def main():
    pdf_path = "document_qa/documents/general-accident-financial-statements.pdf"

    text = extract_text(pdf_path=pdf_path, pages=(0, 5))

    ts = TextSplitter(text=text, preprocessing=True)
    docs = ts.st_split_by_token(tokens_per_chunk=384, chunk_overlap=30)
    db = create_db(documents=docs)
    query = "What did the president say about Ketanji Brown Jackson"
    docs = db.similarity_search(query)
    print(docs[0].page_content)


if __name__ == "__main__":
    main()
