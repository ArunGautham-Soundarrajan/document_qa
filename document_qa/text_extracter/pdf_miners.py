from io import BytesIO
from pathlib import Path
from typing import Callable, Dict, List

import fitz
from vector_search.text_embedders import TextEmbedding


def extract_text(
    file_name: str,
    file_content: bytes,
    doc_id: str,
    text_splitter: Callable,
    params: Dict,
    embedder: TextEmbedding,
    pages: tuple[int, int] | None = None,
) -> List[Dict]:
    """Extract, chunk it, vectorise the text from a PDF

    :param str file_name: name of the pdf document
    :param bytes file_content: pdf document object
    :param str doc_id: uuid for the document
    :param Callable text_splitter: Langchain text splitter function
    :param Dict params: Any parameters for the text splitter function
    :param TextEmbedding embedder: TextEmbedder object to vectorise the text
    :param tuple[int, int] | None pages: Page number slice to extract, defaults to None
    :return List[Dict]: dictionary of each chunk with its embeddings, and other meta data
    """

    extracted_text = []
    with fitz.open(filetype="pdf", stream=BytesIO(file_content)) as doc:
        if pages:
            selected_pages = doc.pages(*pages)
        else:
            selected_pages = doc

        for idx, page in enumerate(selected_pages):
            text = page.get_text("text")

            splitted_text = text_splitter(text=text, **params)
            for doc in splitted_text:
                extracted_text.append(
                    {
                        "doc_id": doc_id,
                        "text": doc,
                        "page_number": pages[0] + idx if pages else idx + 1,
                        "document_name": file_name,
                        "embeddings": embedder.embed_documents(doc),
                    }
                )

    return extracted_text
