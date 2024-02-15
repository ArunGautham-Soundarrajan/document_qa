from functools import partial
from typing import List

from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter,
    TokenTextSplitter,
)
from langchain_core.documents import Document
from textacy import preprocessing
from transformers import GPT2TokenizerFast


class TextSplitter:
    def __init__(self, preprocessing: bool = True) -> None:
        self.preprocessing = preprocessing

    def normalizing_text(self, text: str) -> str:
        preproc = preprocessing.make_pipeline(
            preprocessing.normalize.hyphenated_words,
            preprocessing.normalize.quotation_marks,
            partial(preprocessing.normalize.repeating_chars, chars=".", maxn=2),
            partial(preprocessing.normalize.repeating_chars, chars=",", maxn=2),
            partial(preprocessing.normalize.repeating_chars, chars=" ", maxn=2),
            preprocessing.normalize.unicode,
            preprocessing.normalize.whitespace,
            preprocessing.remove.html_tags,
        )

        try:
            return preproc(text)
        except Exception as e:
            raise e

    def split_by_char(
        self,
        text: str,
        chunk_size: int,
        chunk_overlap: int,
        is_separator_regex: bool = False,
    ) -> List[str]:
        if preprocessing:
            text = self.normalizing_text(text)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=is_separator_regex,
        )

        return text_splitter.split_text(text)

    def st_split_by_token(
        self,
        text: str,
        tokens_per_chunk: int,
        chunk_overlap: int,
        model_name: str = "sentence-transformers/all-mpnet-base-v2",
    ) -> List[str]:
        if preprocessing:
            text = self.normalizing_text(text)
        tokens_per_chunk = tokens_per_chunk if tokens_per_chunk <= 384 else 384
        text_splitter = SentenceTransformersTokenTextSplitter(
            model_name=model_name,
            tokens_per_chunk=tokens_per_chunk,
            chunk_overlap=chunk_overlap,
        )

        return text_splitter.split_text(text)

    # TODO: Find a way to change a seperator
    def hf_split_by_token(
        self,
        text: str,
        chunk_size: int,
        chunk_overlap: int,
    ) -> List[str]:
        if preprocessing:
            text = self.normalizing_text(text)
        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            tokenizer=tokenizer,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        return text_splitter.split_text(text)

    def tiktoken_split_by_token(
        self, text: str, chunk_size: int, chunk_overlap: int
    ) -> List[str]:
        if preprocessing:
            text = self.normalizing_text(text)
        text_splitter = TokenTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        return text_splitter.split_text(text)
