import os
import tempfile
from langchain_docling import DoclingLoader
from langchain_opendataloader_pdf import OpenDataLoaderPDFLoader


def load_text_file():
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as temp_file:
        temp_file.write(b"This is a sample text file for testing.")
        temp_file_path = temp_file.name

    try:
        loader = DoclingLoader(temp_file_path)
        docs = loader.load()

        for doc in docs:
            print(doc)

    finally:
        os.remove(temp_file_path)

def load_csv():
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp_file:
        temp_file.write(b"test document")
        temp_file_path = temp_file.name

    try:
        loader = DoclingLoader(temp_file_path)
        docs = loader.load()

        for doc in docs:
            print(doc)

    finally:
        os.remove(temp_file_path)

def load_pdf():
    loader = OpenDataLoaderPDFLoader(
            file_path=["assets/sample.pdf"],
            format="text"
    )
    docs = loader.load()
    print(f"Loaded {len(docs)} pages from the PDF.")
    for doc in docs:
        print(doc.metadata)
        print(f"{doc.page_content[:100]}...\n################################################")  # Print first 100 characters of each page

   
#  load_text_file()
#  load_csv()

load_pdf()
