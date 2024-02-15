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
