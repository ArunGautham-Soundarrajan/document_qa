from fastapi import APIRouter, Depends, HTTPException, Request
from routers.models.requests import SearchItem
from routers.models.response import SearchResult
from routers.v1.dependencies import get_db
from vector_search.milvus_client import Milvus

router = APIRouter(
    prefix="/search",
    responses={404: {"description": "Not found"}},
    tags=["Vector Search"],
)


@router.post("/", response_model=SearchResult)
async def vector_search(
    request: Request, item: SearchItem, vector_db: Milvus = Depends(get_db)
):
    """Perform vector serach and generate LLM response

    :param Request request: Web request object
    :param SearchItem item: Document id and question
    :param Milvus vector_db: Milvus client session, defaults to Depends(get_db)
    :return _type_: Generated response and source for the response
    """
    question_embedding = request.state.text_embedder.embed_query(item.question)
    results = vector_db.vector_search(
        question_embedding=question_embedding, doc_id=item.doc_id
    )

    # Get the context and source from the returned results
    source, context = request.state.text_embedder.get_context_source(entities=results)

    # Call the LLM to generate the answer
    generated_response = request.state.text_embedder.generate_response(
        question=item.question, context=context
    )

    return SearchResult(generated_answer=generated_response, sources=source)
