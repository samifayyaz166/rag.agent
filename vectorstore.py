from pathlib import Path

from langchain_chroma import Chroma


# Keep the database location stable at the project root.
PROJECT_DIR = Path(__file__).resolve().parent
VECTORSTORE_DIR = PROJECT_DIR / "vectorstore"


def create_vectorstore(documents, embeddings):
    """
    Open the existing persistent Chroma database.
    Add the initial documents only when the collection is empty.
    """

    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)

    vectorstore = Chroma(
        persist_directory=str(VECTORSTORE_DIR),
        embedding_function=embeddings,
    )

    # Avoid re-embedding the same initial corpus on every app startup.
    if vectorstore._collection.count() == 0 and documents:
        vectorstore.add_documents(documents)

    return vectorstore
