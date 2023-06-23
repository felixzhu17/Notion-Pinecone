import hashlib
from tqdm import tqdm
import os
import re


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


def format_notion_source(string):
    """
    Formats the input string by replacing backslashes with forward slashes, removing UUID-like sequences and trailing spaces.

    The function expects the UUID-like sequences to be 32 alphanumeric characters long. These sequences are removed if they
    are followed by a ".md", a slash "/", or are sandwiched between spaces. All trailing spaces at the end of the string are
    also removed.

    Args:
        string (str): The input string to be formatted. The string is expected to contain UUID-like sequences and possibly
        backslashes and trailing spaces.

    Returns:
        str: The formatted string with UUID-like sequences, backslashes and trailing spaces removed.
    """
    string = string.replace("\\", "/")
    cleaned_string = re.sub(
        r"([a-f0-9]{32}/|[a-f0-9]{32}\.md| [a-f0-9]{32} )", "", string
    )
    cleaned_string = cleaned_string.rstrip()
    return cleaned_string
