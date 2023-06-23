import hashlib
from tqdm import tqdm

def chunk_data_sources(data, sources, text_split_func):
    docs = []
    metadatas = []
    for i, d in tqdm(enumerate(data)):
        splits = text_split_func(d)
        docs.extend(splits)
        metadatas.extend([{"text": split, "source": sources[i]} for split in splits])
    return docs, metadatas


def hash(text):
    return hashlib.sha256(text.encode()).hexdigest()


def hash_docs(docs):
    return [hash(doc) for doc in docs]
