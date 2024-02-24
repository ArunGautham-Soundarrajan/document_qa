from typing import List, Dict, Tuple

import numpy as np
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from openai import OpenAI
from config import settings


class TextEmbedding:
    """Text Embedder to vectorise text and to generate response"""

    def __init__(self) -> None:
        self.model = HuggingFaceInstructEmbeddings(
            query_instruction="Represent the query for retrieval: "
        )
        self.open_ai = OpenAI(api_key=settings.open_ai.api_key)

    def embed_documents(self, text: List[str]) -> List[float]:
        """Vectorise a chunk of text

        :param List[str] text: Chunk of text to vectorise
        :return List[float]: Vectorised text
        """
        return np.ravel(self.model.embed_documents([text])).tolist()

    def embed_query(self, query: str) -> List[float]:
        """Vectorise the query text

        :param str query: Query text to vectorise
        :return List[float]: Vectorised query
        """
        return [self.model.embed_query(query)]

    def get_context_source(self, entities: List[Dict]) -> Tuple[List[Dict], str]:
        """Get the source and context from the given Dictionary

        :param List[Dict] entities: Output from Milvus vector search
        :return Tuple[List[Dict], str]: Source and context
        """
        sources = []
        context = ""
        for entity in entities[0]:
            sources.append(
                f"{entity['entity']['document_name']}, page number {entity['entity']['page_number']}"
            )
            context += entity["entity"]["text"]

        return list(sorted(set(sources))), context

    def generate_response(self, question: str, context: str) -> str:
        """_summary_

        :param str question: Query text
        :param str context: Relevant context to the query
        :return str: Generated response
        """
        completion = self.open_ai.chat.completions.create(
            model=settings.open_ai.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a Question answering agent, skilled in answering questions from the context provided.\
                        If the answer doesn't exist in the context, you would say 'Couldn't find the answer in the document'",
                },
                {"role": "user", "content": f"{question} \n\n {context} "},
            ],
        )

        return completion.choices[0].message.content
