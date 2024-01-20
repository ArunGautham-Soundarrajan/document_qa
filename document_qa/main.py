from document_qa.text_extracter.pdf_miners import extract_text
from document_qa.vector_search.text_cleaning import normalizing_text


def main():
    pdf_path = "document_qa/documents/general-accident-financial-statements.pdf"
    text = extract_text(pdf_path=pdf_path, pages=(0, 4))
    print("Text before cleaning")
    print(text)

    cleaned_text = normalizing_text(text=text)
    print("Cleaned Text")
    print(cleaned_text)


if __name__ == "__main__":
    main()
