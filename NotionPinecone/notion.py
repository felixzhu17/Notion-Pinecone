from pathlib import Path
from .utils import *
from .pinecone import PineconeVectorStore
import os
from tqdm import tqdm
import pdfplumber
from docx import Document


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
        llm_model="gpt-3.5-turbo",
        proxy_connection=None,
    ):
        """The constructor for NotionPinecone class."""
        self.notion_path = notion_path
        self.embedder = embedder
        self.markdown_paths = list(Path(notion_path).glob("**/*.md"))
        self.pdf_paths = list(Path(notion_path).glob("**/*.pdf"))
        self.doc_paths = list(Path(notion_path).glob("**/*.doc")) + list(Path(notion_path).glob("**/*.docx"))
        self.docs, self.metadata = self.extract_notion_data()
        self.ids = hash_docs(self.docs)
        self.unique_notion_pages = self.get_unique_notion_pages()
        PineconeVectorStore.__init__(
            self,
            database_name=database_name,
            embedder=embedder,
            metadata_text_field=metadata_text_field,
            openai_api_key=openai_api_key,
            pinecone_api_key=pinecone_api_key,
            pinecone_environment=pinecone_environment,
            llm_model=llm_model,
            proxy_connection=proxy_connection,
        )

    def extract_notion_data(self):
        """
        Extract data from Notion files.

        Returns:
            tuple: Tuple containing chunked data and sources.
        """
        
        print("Reading Doc files...")
        doc_docs, doc_sources = self.extract_doc(self.doc_paths)
        print("Reading PDF files...")
        pdf_docs, pdf_sources = self.extract_pdf(self.pdf_paths)
        print("Reading MD files...")
        md_docs, md_sources = self.extract_md(self.markdown_paths)
        
        data = pdf_docs + doc_docs + md_docs
        sources = pdf_sources + doc_sources + md_sources
        
        print("Chunking files...")
        return chunk_data_sources(data, sources, self.embedder.split_text)

    def extract_pdf(self, paths):
        data = []
        sources = []
        for p in tqdm(paths):
            try:
                with pdfplumber.open(p) as pdf:
                    content = ""
                    for page in pdf.pages:
                        content += page.extract_text()
                    data.append(content)
                    sources.append(format_notion_source(str(p)))
            except Exception as e:
                print(f"Unable to read file: {p}. Exception: {str(e)}. Skipping...")
        return data, sources
                
    def extract_doc(self, paths):
        data = []
        sources = []
        for p in tqdm(paths):
            try:
                doc = Document(p)
                content = ' '.join([paragraph.text for paragraph in doc.paragraphs])
                data.append(content)
                sources.append(format_notion_source(str(p)))
            except Exception as e:
                print(f"Unable to read file: {p}. Exception: {str(e)}. Skipping...")
        return data, sources

    def extract_md(self, paths):
        data = []
        sources = []
        for p in tqdm(paths):
            try:
                with open(p, encoding="utf-8") as f:
                    data.append(f.read())
                    sources.append(format_notion_source(str(p)))
            except UnicodeDecodeError:
                print(f"Unable to read file: {p}. Skipping...")
        return data, sources

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
        page_does_not_exist_in_db = [
            i
            for i in tqdm(self.unique_notion_pages)
            if not self._check_notion_page_in_db(i)
        ]
        filtered = [
            (id, doc, meta)
            for id, doc, meta in zip(self.ids, self.docs, self.metadata)
            if meta["source"] in page_does_not_exist_in_db
        ]

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
        return (
            len(
                self.query(
                    [0] * self.embedder.dimensions,
                    top_k=1,
                    filter={"source": notion_page},
                )["matches"]
            )
            > 0
        )
