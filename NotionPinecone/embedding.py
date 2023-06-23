import os
import tiktoken
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from .utils import *
import warnings

# Overlap factor for text splitting
OVERLAP_FACTOR = 20


class Embedder:
    """
    A class for text tokenization and embedding.

    Attributes:
        name (str): The name of the model used for embeddings.
        tokenizer (str): The tokenizer to be used.
        dimensions (int): The dimensions for the embedding.
        max_tokens (int): The maximum number of tokens allowed.
        overlap_factor (int): The factor of overlap in text splitting.
        openai_api_key (str): The key for the OpenAI API.
    """

    def __init__(
        self,
        name,
        tokenizer,
        dimensions,
        max_tokens,
        overlap_factor=OVERLAP_FACTOR,
        openai_api_key=None,
    ):
        """The constructor for Embedder class."""
        self.name = name
        self.tokenizer = tokenizer
        self.dimensions = dimensions
        self.max_tokens = max_tokens
        self.overlap_factor = overlap_factor
        self.openai_api_key = get_env_var("OPENAI_API_KEY", openai_api_key)

    def encode(self, text):
        """
        Encodes a given text.

        Args:
            text (str): The text to encode.

        Returns:
            list: The encoded text.
        """
        return self.encoder.encode(text, disallowed_special=())

    def token_length(self, text):
        """
        Determines the token length of a given text.

        Args:
            text (str): The text to evaluate.

        Returns:
            int: The token length of the text.
        """
        tokens = self.encode(text)
        return len(tokens)

    def split_text(self, text):
        """
        Splits a given text.

        Args:
            text (str): The text to split.

        Returns:
            list: The split text.
        """
        return self.text_splitter.split_text(text)

    def embed_documents(self, documents):
        """
        Embeds a given list of documents.

        Args:
            documents (list): The documents to embed.

        Returns:
            list: The embedded documents.
        """
        return self.embedder.embed_documents(documents)

    def embed(self, text):
        """
        Embeds a given text.

        Args:
            text (str): The text to embed.

        Returns:
            list: The embedded text.
        """
        return self.embedder.embed_query(text)

    @property
    def text_splitter(self):
        """Defines the text splitter using the RecursiveCharacterTextSplitter."""
        return RecursiveCharacterTextSplitter(
            chunk_size=self.max_tokens,
            chunk_overlap=self.max_tokens // self.overlap_factor,
            length_function=self.token_length,
            separators=["\n\n", "\n", " ", ""],
        )

    @property
    def encoder(self):
        """Defines the encoder using tiktoken."""
        return tiktoken.get_encoding(self.tokenizer)

    @property
    def embedder(self):
        """Defines the embedder using the OpenAIEmbeddings."""
        return OpenAIEmbeddings(model=self.name, openai_api_key=self.openai_api_key)


# Instance of the Embedder class using the model 'text-embedding-ada-002'
def ada_v2(openai_api_key=None):
    return Embedder(
        name="text-embedding-ada-002",
        tokenizer="cl100k_base",
        dimensions=1536,
        max_tokens=500,
        openai_api_key=openai_api_key,
    )
