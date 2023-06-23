import hashlib
from tqdm import tqdm
import os

def get_env_var(key, var=None):
    """Retrieve an environment variable or return a given default value.

    Raises:
        EnvironmentError: If the environment variable doesn't exist and no default value is provided.

    Args:
        key (str): The environment variable key.
        var (str, optional): The default value to return if the environment variable is not set.

    Returns:
        str: The value of the environment variable or the default value.
    """
    var = var or os.getenv(key)
    if var is None:
        raise EnvironmentError(f"Environment variable {key} is not set")
    return var


def chunk_data_sources(data, sources, text_split_func):
    """
    Splits data into chunks and associates each chunk with its source.

    Args:
        data (list): The list of data to be chunked.
        sources (list): The list of sources for each data item.
        text_split_func (callable): A function that splits a text into chunks.

    Returns:
        list, list: Lists of chunks and their corresponding metadata.
    """
    docs = []
    metadatas = []
    for i, d in tqdm(enumerate(data)):
        splits = text_split_func(d)
        docs.extend(splits)
        metadatas.extend([{"text": split, "source": sources[i]} for split in splits])
    return docs, metadatas


def hash(text):
    """
    Computes the SHA-256 hash of a text.

    Args:
        text (str): The text to be hashed.

    Returns:
        str: The hash of the text.
    """
    return hashlib.sha256(text.encode()).hexdigest()


def hash_docs(docs):
    """
    Computes the hash of each document in a list.

    Args:
        docs (list): The list of documents to be hashed.

    Returns:
        list: The list of hashes for each document.
    """
    return [hash(doc) for doc in docs]
