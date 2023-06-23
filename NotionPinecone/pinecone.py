import pinecone
import os
from langchain.vectorstores import Pinecone
from tqdm import tqdm
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQAWithSourcesChain

LLM_MODEL = "gpt-3.5-turbo"

class PineconeVectorStore(Pinecone, pinecone.GRPCIndex):
    def __init__(
        self,
        database_name,
        embedder,
        metadata_text_field="text",
        openai_api_key = os.environ.get("OPENAI_API_KEY"),
        pinecone_api_key=os.environ.get("PINECONE_API_KEY"),
        pinecone_environment=os.environ.get("PINECONE_ENVIRONMENT"),
        llm_model = "gpt-3.5-turbo"
    ):
        self.database_name = database_name
        self.embedder = embedder
        self.metadata_text_field = metadata_text_field
        self.openai_api_key=openai_api_key
        self.pinecone_api_key = pinecone_api_key
        self.pinecone_environment = pinecone_environment
        self.llm_model = llm_model
        pinecone.init(
            api_key=self.pinecone_api_key, environment=self.pinecone_environment
        )
        pinecone.GRPCIndex.__init__(self, self.database_name)
        Pinecone.__init__(
            self,
            pinecone.Index(self.database_name),
            self.embedder.embed,
            self.metadata_text_field,
        )

    def create_database(self):
        if self.database_name not in pinecone.list_indexes():
            pinecone.create_index(
                name=self.database_name,
                metric="cosine",
                dimension=self.embedder.dimensions,
            )

    def upload_in_batches(self, ids, embeddings, metadata, batch_size=100):
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
        return ChatOpenAI(
            openai_api_key=self.openai_api_key,
            model_name=self.llm_model,
            temperature=0.0
        )

    @property
    def question_answer(self):
        return RetrievalQAWithSourcesChain.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.as_retriever()
        )

    def ask(self, query):
        return self.question_answer(query)



        