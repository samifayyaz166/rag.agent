from pathlib import Path

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
    CSVLoader,
    WebBaseLoader,
)

from langchain_core.documents import Document
import pandas as pd


def load_file(file_path):

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"File not found: {file_path}"
        )

    extension = path.suffix.lower()

    print(f"Loading: {path.name}")

    # PDF
    if extension == ".pdf":
        loader = PyPDFLoader(str(path))
        return loader.load()

    # Word
    elif extension == ".docx":
        loader = Docx2txtLoader(str(path))
        return loader.load()

    # TXT
    elif extension == ".txt":
        loader = TextLoader(
            str(path),
            encoding="utf-8"
        )
        return loader.load()

    # CSV
    elif extension == ".csv":
        loader = CSVLoader(str(path))
        return loader.load()

    # Excel
    elif extension in [".xlsx", ".xls"]:

        dataframe = pd.read_excel(path)

        documents = []

        for index, row in dataframe.iterrows():

            row_text = "\n".join(
                f"{column}: {value}"
                for column, value in row.items()
            )

            documents.append(
                Document(
                    page_content=row_text,
                    metadata={
                        "source": str(path),
                        "file_type": "excel",
                        "row": index + 1
                    }
                )
            )

        return documents

    else:
        raise ValueError(
            f"Unsupported file type: {extension}"
        )


def load_webpage(url):
    print(f"Loading webpage: {url}")

    loader = WebBaseLoader(url)

    return loader.load()
