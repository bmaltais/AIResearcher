import os
import hashlib
import argparse
from typing import List

import PyPDF2
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
import markdown
from docx import Document as DocxDocument
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

# Configuration
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200
MODEL = "nomic-embed-text"

# Initialize Ollama embeddings
embeddings = OllamaEmbeddings(model=MODEL)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Process documents and store in ChromaDB")
    parser.add_argument("--collection", type=str, default="default_collection", help="Name of the ChromaDB collection")
    parser.add_argument("--document-directory", type=str, default="./documents", help="Directory containing documents to process")
    parser.add_argument("--chroma-db-path", type=str, default="./chroma_db", help="Path to store ChromaDB")
    return parser.parse_args()

def calculate_file_hash(file_path: str) -> str:
    with open(file_path, "rb") as file:
        return hashlib.md5(file.read()).hexdigest()

def document_exists(db, file_path: str) -> bool:
    file_hash = calculate_file_hash(file_path)
    results = db.get(where={"source": file_path})
    return any(metadata.get('file_hash') == file_hash for metadata in results.get('metadatas', []))

def extract_text_from_docx(file_path: str) -> str:
    doc = DocxDocument(file_path)
    return "\n".join([paragraph.text for paragraph in doc.paragraphs])

def extract_text(file_path: str) -> str:
    _, ext = os.path.splitext(file_path)
    if ext == ".pdf":
        with open(file_path, "rb") as file:
            return "\n".join(page.extract_text() for page in PyPDF2.PdfReader(file).pages)
    elif ext == ".epub":
        book = epub.read_epub(file_path)
        return "\n".join(BeautifulSoup(item.get_content(), "html.parser").get_text()
                         for item in book.get_items() if item.get_type() == ebooklib.ITEM_DOCUMENT)
    elif ext == ".md":
        with open(file_path, "r", encoding="utf-8") as file:
            return BeautifulSoup(markdown.markdown(file.read()), "html.parser").get_text()
    elif ext == ".txt":
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()
    elif ext == ".docx":
        return extract_text_from_docx(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

def chunk_text(text: str) -> List[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, length_function=len
    )
    return text_splitter.create_documents([text])

def cleanup_chromadb(db, document_directory: str):
    all_docs = db.get()
    existing_files = {os.path.abspath(os.path.join(root, file))
                      for root, _, files in os.walk(document_directory)
                      for file in files}

    docs_to_remove = [doc_id for doc_id, metadata in zip(all_docs['ids'], all_docs['metadatas'])
                      if 'source' in metadata and os.path.abspath(metadata['source']) not in existing_files]

    if docs_to_remove:
        db.delete(ids=docs_to_remove)
        print(f"Removed {len(docs_to_remove)} documents from ChromaDB that no longer exist in the file system.")
    else:
        print("No documents to remove from ChromaDB.")

    print(f"Total documents remaining in ChromaDB: {len(db.get()['ids'])}")

def process_file(db, file_path: str):
    if document_exists(db, file_path):
        print(f"Document {file_path} already exists in the database. Skipping.")
        return

    print(f"Processing {os.path.basename(file_path)}")
    text = extract_text(file_path)
    chunks = chunk_text(text)
    file_hash = calculate_file_hash(file_path)

    updated_chunks = [
        Document(page_content=chunk.page_content,
                 metadata={**chunk.metadata, "file_hash": file_hash, "source": file_path, "chunk": i})
        for i, chunk in enumerate(chunks)
    ]

    db.add_documents(updated_chunks)
    print(f"Processed {os.path.basename(file_path)}")

def process_documents_to_chroma(db, directory_path: str):
    for root, _, files in os.walk(directory_path):
        for file in files:
            process_file(db, os.path.join(root, file))

    cleanup_chromadb(db, directory_path)
    print("All documents have been processed and inserted into ChromaDB.")

if __name__ == "__main__":
    args = parse_arguments()
    
    # Initialize ChromaDB with the specified collection
    db = Chroma(persist_directory=args.chroma_db_path, embedding_function=embeddings, collection_name=args.collection)
    
    process_documents_to_chroma(db, args.document_directory)