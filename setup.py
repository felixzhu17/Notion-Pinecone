from setuptools import find_packages, setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="notion-pinecone",
    version="0.0.24",
    license="MIT",
    author="Felix Zhu",
    author_email="zhu.felix@outlook.com",
    description="Automated QA of Notion in Pinecone",
    long_description=long_description,
    packages=find_packages(),
    setup_requires=["setuptools_scm"],
    url="https://github.com/felixzhu17/Notion-Pinecone",
    install_requires=[
        "langchain",
        "openai",
        "tiktoken",
        "pinecone-client[grpc]"
    ],
)