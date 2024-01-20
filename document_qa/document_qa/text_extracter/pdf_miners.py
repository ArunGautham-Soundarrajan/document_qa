import fitz


def extract_text(pdf_path: str, pages: tuple[int, int] | None = None) -> str:
    extracted_text = ""
    with fitz.open(pdf_path) as doc:
        if pages:
            selected_pages = doc.pages(*pages)
        else:
            selected_pages = doc

        for page in selected_pages:
            extracted_text += page.get_text("text")

    return extracted_text
