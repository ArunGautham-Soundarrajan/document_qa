from text_extracter.pdf_miners import extract_text
from vector_search.text_cleaning import normalizing_text
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceInstructEmbeddings


def create_db(pdf_path: str, pages=tuple[int, int]) -> Chroma:
    # Extract the text from the pdf
    text = extract_text(pdf_path=pdf_path, pages=pages)
    cleaned_text = normalizing_text(text=text)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=20,
        length_function=len,
        is_separator_regex=False,
    )
    pages = text_splitter.create_documents([cleaned_text])

    # Load the embedding model
    embeddings = HuggingFaceInstructEmbeddings(
        query_instruction="Represent the query for retrieval: "
    )
    db = Chroma.from_documents(pages, embeddings)
    return db
