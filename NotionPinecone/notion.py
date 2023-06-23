from pathlib import Path
from .utils import *
from .pinecone import PineconeVectorStore
import os
from tqdm import tqdm

class NotionPinecone(PineconeVectorStore):
    """
    A class that inherits from PineconeVectorStore and is used to manage Notion data in Pinecone.

    Attributes:
        notion_path (str): Path to Notion files.
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
        notion_path,
        database_name,
        embedder,
        metadata_text_field="text",
        openai_api_key=None,
        pinecone_api_key=None,
        pinecone_environment=None,
        llm_model = "gpt-3.5-turbo"
    ):
        """The constructor for NotionPinecone class."""
        self.notion_path = notion_path
        self.embedder = embedder
        self.notion_file_paths = list(Path(notion_path).glob("**/*.md"))
        self.docs, self.metadata = self.extract_notion_data()
        self.ids = hash_docs(self.docs)
        self.unique_notion_pages = self.get_unique_notion_pages()
        PineconeVectorStore.__init__(
            self,
            database_name=database_name,
            embedder=embedder,
            metadata_text_field=metadata_text_field,
            openai_api_key = openai_api_key,
            pinecone_api_key=pinecone_api_key,
            pinecone_environment=pinecone_environment,
            llm_model = llm_model
        )

    def extract_notion_data(self):
        """
        Extract data from Notion files.

        Returns:
            tuple: Tuple containing chunked data and sources.
        """
        data = []
        sources = []
        print("Reading Notion files...")
        for p in tqdm(self.notion_file_paths):
            with open(p, encoding="utf-8") as f:
                try:
                    data.append(f.read())
                    sources.append(str(p))
                except UnicodeDecodeError:
                    print(f"Unable to read file: {p}. Skipping...")
        print("Chunking Notion files...")
        return chunk_data_sources(data, sources, self.embedder.split_text)

    def get_unique_notion_pages(self):
        """
        Get unique Notion pages.

        Returns:
            list: List of unique Notion pages.
        """
        return list(set([i["source"] for i in self.metadata]))

    def upload_notion_vectors(self):
        """
        Upload Notion pages to Pinecone. If all pages are already in Pinecone, no upload is performed.
        """
        ids, docs, metadata = self._filter_vectors_not_in_database()
        if len(ids) == 0:
            print("All Notion pages are already in Pinecone. No need to upload.")
            return
        else:
            print(f"Uploading {len(ids)} Notion pages to Pinecone...")
            self.upload_in_batches(ids, self.embedder.embed_documents(docs), metadata)
            print("Done.")

    def _filter_vectors_not_in_database(self):
        """
        Check if pages are in Pinecone and filter those that are not.

        Returns:
            tuple: Tuple containing IDs, documents and metadata of the pages not in Pinecone.
        """
        print("Checking if pages are in Pinecone...")
        page_does_not_exist_in_db = [i for i in tqdm(self.unique_notion_pages) if not self._check_notion_page_in_db(i)]
        filtered = [(id, doc, meta) for id, doc, meta in zip(self.ids, self.docs, self.metadata) if meta['source'] in page_does_not_exist_in_db]

        if filtered:
            filtered_ids, filtered_docs, filtered_metas = zip(*filtered)
            return list(filtered_ids), list(filtered_docs), list(filtered_metas)
        else:
            return [], [], []

    def _check_notion_page_in_db(self, notion_page):
        """
        Check if a Notion page is in the Pinecone database.

        Args:
            notion_page (str): The Notion page to check.

        Returns:
            bool: True if the page is in the database, False otherwise.
        """
        return len(self.query([0] * self.embedder.dimensions, top_k=1, filter={"source": notion_page})['matches']) > 0
