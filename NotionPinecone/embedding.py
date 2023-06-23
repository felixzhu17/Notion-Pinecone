import tiktoken
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
import os

OVERLAP_FACTOR = 20


class Embedder:
    def __init__(
        self,
        name,
        tokenizer,
        dimensions,
        max_tokens,
        overlap_factor=OVERLAP_FACTOR,
        openai_api_key=os.environ.get("OPENAI_API_KEY"),
    ):
        self.name = name
        self.tokenizer = tokenizer
        self.dimensions = dimensions
        self.max_tokens = max_tokens
        self.overlap_factor = overlap_factor
        self.openai_api_key = openai_api_key

    def encode(self, text):
        return self.encoder.encode(text, disallowed_special=())

    def token_length(self, text):
        tokens = self.encode(text)
        return len(tokens)

    def split_text(self, text):
        return self.text_splitter.split_text(text)

    def embed_documents(self, documents):
        return self.embedder.embed_documents(documents)

    def embed(self, text):
        return self.embedder.embed_query(text)

    @property
    def text_splitter(self):
        return RecursiveCharacterTextSplitter(
            chunk_size=self.max_tokens,
            chunk_overlap=self.max_tokens // self.overlap_factor,
            length_function=self.token_length,
            separators=["\n\n", "\n", " ", ""],
        )

    @property
    def encoder(self):
        return tiktoken.get_encoding(self.tokenizer)

    @property
    def embedder(self):
        return OpenAIEmbeddings(model=self.name, openai_api_key=self.openai_api_key)


ADA_V2 = Embedder(
    name="text-embedding-ada-002",
    tokenizer="cl100k_base",
    dimensions=1536,
    max_tokens=500,
)
