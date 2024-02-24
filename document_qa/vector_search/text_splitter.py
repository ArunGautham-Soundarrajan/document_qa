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
    """Langchain text splitters"""

    def __init__(self, preprocessing: bool = True) -> None:
        self.preprocessing = preprocessing

    def normalizing_text(self, text: str) -> str:
        """Preprocess the input text

        :param str text: Text to preprocess
        :return str: Preprocessed and normalised text
        """
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
        """Split the text by character

        :param str text: Text to split
        :param int chunk_size: Size of each chunk
        :param int chunk_overlap: Overlap between each chunks
        :param bool is_separator_regex: _description_, defaults to False
        :return List[str]: List of chunks
        """
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
        """Split text by token using sentence transformer model

        :param str text: Text to split
        :param int tokens_per_chunk: Number of tokens per chunk
        :param int chunk_overlap: Overlap between each chunks
        :param str model_name: Name of the sentencetransformer model, defaults to "sentence-transformers/all-mpnet-base-v2"
        :return List[str]: List of chunks
        """
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
        """Split text by token using Huggingface model

        :param str text: Text to split
        :param int chunk_size: Size of each chunk
        :param int chunk_overlap: Overlap between each chunks
        :return List[str]: List of chunks
        """
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
        """Split the text using tiktoken which is for Open AI models

        :param str text: Text to split
        :param int chunk_size: Size of each chunk
        :param int chunk_overlap: Overlap between each chunks
        :return List[str]: List of chunks
        """
        if preprocessing:
            text = self.normalizing_text(text)
        text_splitter = TokenTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        return text_splitter.split_text(text)
