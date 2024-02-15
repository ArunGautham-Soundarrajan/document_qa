from database.crud import create_document_item, get_all_documents
from database.schemas import DocumentBase, Item
from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile
from routers.v1.dependencies import get_db, get_sql_db
from sqlalchemy.orm import Session
from text_extracter.pdf_miners import extract_text
from vector_search.milvus_client import Milvus
from vector_search.text_splitter import TextSplitter

router = APIRouter(
    prefix="/files",
    responses={404: {"description": "Not found"}},
)


@router.get("/", response_model=list[Item])
async def list_files(db: Session = Depends(get_sql_db)):
    documents = get_all_documents(db=db)
    return documents


@router.post("/", response_model=Item)
async def upload_file(
    request: Request,
    file: UploadFile,
    vector_db: Milvus = Depends(get_db),
    db: Session = Depends(get_sql_db),
):
    # Store the document information in SQL
    item = DocumentBase(file_name=file.filename)
    result = create_document_item(db=db, item=item)

    # Extract text and store embeddings in vector db
    pdf_content: bytes = await file.read()
    ts = TextSplitter(preprocessing=True)
    text_splitter = ts.tiktoken_split_by_token
    params = {"chunk_size": 300, "chunk_overlap": 50}
    text = extract_text(
        file_name=file.filename,
        file_content=pdf_content,
        doc_id=result.id,
        text_splitter=text_splitter,
        params=params,
        embedder=request.state.text_embedder,
    )
    res = vector_db.insert_to_collection(data=text)

    return result


@router.delete("/{doc_id}")
async def delete_file(
    doc_id: str, vector_db: Milvus = Depends(get_db), db: Session = Depends(get_sql_db)
):
    try:
        vector_db.delete_entity(doc_id=doc_id)
        return {"message": "success"}
    except Exception:
        return HTTPException(status_code=500)
