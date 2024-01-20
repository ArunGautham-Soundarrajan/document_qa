import fitz
from pprint import pprint

pdf_path = "document_qa/documents/general-accident-financial-statements.pdf"


def extract_text(pdf_path: str, pages: tuple[int, int] | None = None) -> dict:
    extracted_text = {}
    with fitz.open(pdf_path) as doc:
        if pages:
            selected_pages = doc.pages(*pages)
        else:
            selected_pages = doc

        for page in selected_pages:
            extracted_text[page.number] = page.get_text("text")

    return extracted_text


pprint(extract_text(pdf_path=pdf_path, pages=(0, 4)))
