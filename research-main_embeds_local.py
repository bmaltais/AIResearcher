import chromadb
import importlib
import sys
import os
import argparse
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from datetime import datetime

from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

load_dotenv()

QUERY_FILE = 'queries.txt'
CONVERSATION_FILE = './Research.md'
MODEL = "gemma2"

N_RESULTS = 5

SYSTEM_INSTRUCTION = """
You are an experienced research assistant. You will answer with detailed, lengthy university level answers.

Answer the question based on the provided context. If the context doesn't contain enough information to answer the question fully, say so and provide the best answer you can with the available information.

Always provide your answer in markdown.
"""

IMPROVE_INSTRUCTION = """
You are an expert question refiner and context analyzer. Your role is to:

1. Take a user's original question
2. Analyze the provided contextual information
3. Reword and improve the original question with more pertinent words from the provided context

Follow these guidelines:

- Create an improved question that make use of more words from the provided context. Make sure to retain the exact spirit of the original question and that it mean the same thing
- Do not provide the answer as part of the expanded question
- Incorporate relevant details and exact words from the context into the expanded question but DO NOT use or refer to the word context itself
- If the context lacks sufficient information, simply return the original question

Always begin your response with your expanded question version and only provide the expanded question version.

DO NOT provide any information about what lead to the improved question. Just provide the improved question. Nothing else.
"""

N_HISTORY_LINES = 10

def setup_argparse():
    parser = argparse.ArgumentParser(description="ChromaDB and Ollama Query Script")
    parser.add_argument("--collection", type=str, default="default_collection", help="Name of the ChromaDB collection")
    parser.add_argument("--chroma-db-path", type=str, default="./chroma_db", help="Path to store ChromaDB")
    return parser.parse_args()

def setup_embeddings_and_db(chroma_db_path, collection_name:str = "default_collection"):
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    db = Chroma(persist_directory=chroma_db_path, embedding_function=embeddings, collection_name=collection_name)
    return db

def query_chromadb(db, query_text, num_results=20):
    results = db.similarity_search_with_score(query_text, k=num_results)
    processed_results = []
    for doc, score in results:
        processed_results.append({
            "documents": doc.page_content,
            "metadatas": doc.metadata,
            "distances": score
        })
    return processed_results

def setup_ollama(model_name):
    llm = Ollama(model=model_name)
    prompt = PromptTemplate(
        input_variables=["system_instruction", "context", "question"],
        template="System instructions: {system_instruction}\n\nContext: {context}\n\nQuestion: {question}"
    )
    return LLMChain(llm=llm, prompt=prompt)

def query_ollama(chain, context, question, system_instruction):
    response = chain.run(system_instruction=system_instruction, context=context, question=question)
    return response

def log_full_query(query, context, metadata, query_file):
    with open(query_file, 'a', encoding='utf-8') as f:
        f.write(f"--- Query at {datetime.now()} ---\n")
        f.write(f"Question:\n{query}\n\n------------------------------\n")
        f.write(f"Context:\n{context}\n\n")
        f.write("\nMetadata of sources:\n")
        for i, meta in enumerate(metadata, 1):
            f.write(f"Source {i}:")
            for key, value in meta.items():
                f.write(f"  {key}: {value}")

def log_conversation(question, answer, metadata, distances, conversation_file):
    with open(conversation_file, 'a', encoding='utf-8') as f:
        f.write(f"\n---\n## Query at {datetime.now()}\n---\n")
        f.write(f"> [!Question]\n{question}\n\n")
        f.write(f"### Answer\n\n{answer}\n\n")
        f.write("\n```\nMetadata of sources:\n")
        for i, (meta, dist) in enumerate(zip(metadata, distances), 1):
            f.write(f"  Source {i}:")
            for key, value in meta.items():
                f.write(f"    {key}: {value}\n")
            f.write(f"    Distance: {dist:.4f}\n")
        f.write("```\n\n")

def main():
    args = setup_argparse()
    
    db = setup_embeddings_and_db(args.chroma_db_path, args.collection)
    ollama_chain = setup_ollama(MODEL)
    
    improve_enabled = False
    history = ''

    while True:
        if improve_enabled:
            prompt = "\nWhat now? ('quit' to exit, 'forget' to clear history, 'no improve' to disable improvement): "
        else:
            prompt = "\nWhat now? ('quit' to exit, 'forget' to clear history, 'improve' to enable improvement): "

        query = input(prompt)

        if query.lower() == 'quit':
            break
        elif query.lower() == 'forget':
            history = ''
            continue
        elif query.lower() == 'improve':
            improve_enabled = True
            print("Question improvement enabled.")
            continue
        elif query.lower() == 'no improve':
            improve_enabled = False
            print("Question improvement disabled.")
            continue

        results = query_chromadb(db, query, N_RESULTS)
        
        context = "\n\n".join([result['documents'] for result in results])
        metadata = [result['metadatas'] for result in results]
        distances = [result['distances'] for result in results]
        
        log_full_query(query, context, metadata, QUERY_FILE)
        
        if improve_enabled:
            print(f"Improving question with Ollama {MODEL}...")
            improved_question = query_ollama(ollama_chain, context, query, IMPROVE_INSTRUCTION)
            
            print(f"\nImproved question: {improved_question}")
            log_conversation(query, improved_question, metadata, distances, CONVERSATION_FILE)

            query = improved_question
            results = query_chromadb(db, query, N_RESULTS)
            context = "\n\n".join([result['documents'] for result in results])
            metadata = [result['metadatas'] for result in results]
            distances = [result['distances'] for result in results]

        print(f"Querying Ollama {MODEL}...")
        answer = query_ollama(ollama_chain, context, query, SYSTEM_INSTRUCTION)
        
        log_conversation(query, answer, metadata, distances, CONVERSATION_FILE)

        history += '\n' + query + '\n' + answer
        
        print(f"\nOllama {MODEL}'s answer:")
        print(answer)
        print("\nMetadata of sources:")
        for i, (meta, dist) in enumerate(zip(metadata, distances), 1):
            print(f"Source {i}:")
            for key, value in meta.items():
                print(f"  {key}: {value}")
            print(f"  Distance: {dist:.4f}")

if __name__ == "__main__":
    main()