from pydoc import doc
from dotenv import load_dotenv
import os
import tempfile
from pathlib import Path
from langchain_community.document_loaders import (
    TextLoader,
    WebBaseLoader,
    DirectoryLoader,
    PyPDFLoader
) 

from langchain_core.documents import Document


from bs4 import BeautifulSoup
from langchain_core.documents import Document

load_dotenv()



def load_text_file(file_path: str) -> str:


    try:
        # load the text file using TextLoader
        loader = TextLoader(file_path=file_path)
        documents = loader.load() # return Document

        print(f"Loaded {len(documents)} document(s)")
        print(f"Content preview: {documents[0].page_content[:100]}...")
        print(f"Metadata: {documents[0].metadata}")
        # Print the loaded documents
        # for doc in documents:
        #     print("\n\nDocument Content:")
        #     print(doc)
        #     print(doc.page_content)

    except Exception as e:
        print(f"Loading text is failed: {e}")

def web_loader():
    loader = WebBaseLoader("https://en.wikipedia.org/wiki/Web_scraping",
                            bs_kwargs={"parse_only":None})

    documents = loader.load()
    print(f"Loaded: {len(documents)} document(s) from web")
    print(f"Source: {documents[0].metadata.get('source', 'N/A')}")
    print(f"Content length: {len(documents[0].page_content)} characters")
    print(f"Preview: {documents[0].page_content[:200]}...")

"""
def lazy_loader():
    
    with tempfile.TemporaryDirectory() as tempdir:
        # Create sample files
        for i in range(5):
            path = Path(tempdir) / f"doc_{i}.txt"
            path.write_text(f"This is document {i}. It contains sample content.")


    
    loader = DirectoryLoader(tempdir, 
                            glob=".txt", # filter
                            loader_cls=TextLoader)

    
    # Load docs lazely - Very efficient for large dataset
    print(f"Initialized lazy loader for directory: {tempdir}")
    for doc in loader.lazy_load():
        print(f"Document Content Preview: {doc.page_content[:50]}...")

        print(f"Metadata :{doc.metadata['source']}")
"""

# Document Strucutre - This class is used when loader load documents, langchain will create Document object for use (sources, content, etc)
def doc_structure():
    # We can construct our own document
    doc = Document(
        page_content="This is a sample document",
        metadata = {
            "source": "sample_source.txt", 
            "author":"John",
            "length": 30,
                    },
    )

    print("Document Structure")
    print(f"    page_content (type): {type(doc.page_content)}")
    print(f"    page_content: {doc.page_content}")
    print(f"    metadata: {doc.metadata}")

    # Documents are immutable

def pdf_loader(pdf_path: str):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()


    print(f"Loaded {len(documents)} document(s) from PDF")
    for i, doc in enumerate(documents):
        print(f"Document {i+1} Content Preview: {doc.page_content[:100]}")
        print(f"Metadata: {doc.metadata}")





if __name__ == "__main__":
    #load_text_file("C:/Users/mhaddara/OneDrive - Education Above All/Documents/AI Appliction\Production AI Agents/reading.txt")
    #web_loader()
    #lazy_loader()
    #doc_structure()
    pdf_loader("C:/Users/mhaddara/OneDrive - Education Above All/Documents/AI Appliction/Production AI Agents/docs/Cheat-Sheet-Kubernetes.pdf")