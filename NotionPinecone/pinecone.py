import pinecone
from pinecone.core.client.configuration import Configuration as OpenApiConfiguration
import os
from langchain.vectorstores import Pinecone
from tqdm import tqdm
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from .utils import *

LLM_MODEL = "gpt-3.5-turbo"


class PineconeVectorStore(Pinecone, pinecone.GRPCIndex):
    """
    A class that inherits from Pinecone and GRPCIndex to manage Pinecone vector stores.

    Attributes:
        database_name (str): Name of the Pinecone database.
        embedder (obj): An object of a class handling embeddings.
        metadata_text_field (str): Name of the field that holds text data.
        openai_api_key (str): The key for the OpenAI API.
        pinecone_api_key (str): The key for the Pinecone API.
        pinecone_environment (str): The Pinecone environment to be used.
        llm_model (str): The model to be used for language learning.
    """

    def __init__(
        self,
        database_name,
        embedder,
        metadata_text_field="text",
        openai_api_key=None,
        pinecone_api_key=None,
        pinecone_environment=None,
        llm_model=LLM_MODEL,
        proxy_connection=None,
    ):
        """The constructor for PineconeVectorStore class."""
        self.database_name = database_name
        self.embedder = embedder
        self.metadata_text_field = metadata_text_field
        self.openai_api_key = get_env_var("OPENAI_API_KEY", openai_api_key)
        self.pinecone_api_key = get_env_var("PINECONE_API_KEY", pinecone_api_key)
        self.pinecone_environment = get_env_var(
            "PINECONE_ENVIRONMENT", pinecone_environment
        )
        self.llm_model = llm_model
        self.proxy_connection = proxy_connection

        pinecone.init(
            api_key=self.pinecone_api_key,
            environment=self.pinecone_environment,
            openapi_config=self._create_openai_proxy_config()
            if self.proxy_connection
            else None,
        )

        pinecone.GRPCIndex.__init__(self, self.database_name)
        Pinecone.__init__(
            self,
            pinecone.Index(self.database_name),
            self.embedder.embed,
            self.metadata_text_field,
        )

    def create_database(self):
        """Creates a Pinecone database if it doesn't already exist."""
        if self.database_name not in pinecone.list_indexes():
            pinecone.create_index(
                name=self.database_name,
                metric="cosine",
                dimension=self.embedder.dimensions,
            )

    def upload_in_batches(self, ids, embeddings, metadata, batch_size=100):
        """
        Uploads data to Pinecone in batches.

        Args:
            ids (list): The list of IDs.
            embeddings (list): The list of embeddings.
            metadata (list): The list of metadata.
            batch_size (int, optional): The size of each batch. Defaults to 100.
        """
        num_batches = len(ids) // batch_size + (len(ids) % batch_size != 0)

        for i in tqdm(range(num_batches)):
            batch_start = i * batch_size
            batch_end = min((i + 1) * batch_size, len(ids))

            ids_batch = ids[batch_start:batch_end]
            embeddings_batch = embeddings[batch_start:batch_end]
            metadata_batch = metadata[batch_start:batch_end]

            self.upsert(vectors=zip(ids_batch, embeddings_batch, metadata_batch))

    @property
    def llm(self):
        """
        Property that returns a ChatOpenAI object with the specified API key and model.

        Returns:
            ChatOpenAI: A ChatOpenAI object.
        """
        return ChatOpenAI(
            openai_api_key=self.openai_api_key,
            model_name=self.llm_model,
            temperature=0.0,
        )

    @property
    def question_answer(self):
        """
        Property that returns a RetrievalQAWithSourcesChain object.

        Returns:
            RetrievalQAWithSourcesChain: A RetrievalQAWithSourcesChain object.
        """
        return RetrievalQAWithSourcesChain.from_chain_type(
            llm=self.llm, chain_type="stuff", retriever=self.as_retriever()
        )

    def ask(self, query):
        """
        Ask a question and get a response.

        Args:
            query (str): The query to ask.

        Returns:
            str: The response to the query.
        """
        return self.question_answer(query)

    def _create_openai_proxy_config(self):
        openapi_config = OpenApiConfiguration.get_default_copy()
        openapi_config.verify_ssl = True
        openapi_config.proxy = self.proxy_connection
        return openapi_config
