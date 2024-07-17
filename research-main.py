import chromadb
import importlib
import sys
import os
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
import google.generativeai as genai
from datetime import datetime

load_dotenv()

CHROMA_COLLECTION = 'collection'
DB_PATH = "./chromadb"  # Update this to your ChromaDB path

# Google AI Configuration
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
MODEL_NAME = 'gemini-1.5-pro'

# File Paths
QUERY_FILE = 'queries.txt' #Mostly just a log file
CONVERSATION_FILE = './Research.md'

# Query Configuration
N_RESULTS = 20

# System Instruction, play with this a bit maybe customize it to your subject
SYSTEM_INSTRUCTION = """
You are an experienced research assistant. You will answer with detailed, lengthy college level answers, using headings, bold, and bullet points as appropriate for clarity. Answer the question based on the provided context. If the context doesn't contain enough information to answer the question fully, say so and provide the best answer you can with the available information.

Always provide your answer in markdown.
"""

IMPROVE_QUESTION = """
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

# Conversation History Configuration. Not sure this is really helpful.
N_HISTORY_LINES = 10  # Number of lines to read from the conversation history

# def load_config(config_file):
#     # Get the directory of the script
#     script_dir = os.path.dirname(os.path.abspath(__file__))
    
#     # Add the script directory to sys.path
#     sys.path.insert(0, script_dir)
    
#     # Remove the .py extension if present
#     if config_file.endswith('.py'):
#         config_file = config_file[:-3]
    
#     # Import the config module dynamically
#     config = importlib.import_module(config_file)
    
#     # Remove the script directory from sys.path
#     sys.path.pop(0)
    
#     return config

def setup_chroma_client(db_path, collection_name):
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_collection(collection_name)
    return collection

def query_chroma(collection, query, n_results):
    embedding_func = embedding_functions.DefaultEmbeddingFunction()
    
    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        include=["documents", "metadatas", "distances"]
    )
    
    return results

def setup_gemini(model_name, system_instruction, api_key):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name=model_name, safety_settings=None, system_instruction=system_instruction)    
    return model

def query_gemini(model, context, question):
    prompt = f"""Context: \n{context}\n\nQuestion: \n{question}"""

    response = model.generate_content(prompt)
    return response.text

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


def read_last_lines(filename, n_lines):
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            all_lines = file.readlines()
            last_lines = all_lines[-n_lines:]
            history = '\n\n**Conversation History**\n\n'.join(last_lines)
            return history
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def main():
    history = ''
    collection = setup_chroma_client(DB_PATH, CHROMA_COLLECTION)
    gemini_model = setup_gemini(MODEL_NAME, SYSTEM_INSTRUCTION, GOOGLE_API_KEY)
    improve_question_model = setup_gemini(MODEL_NAME, IMPROVE_QUESTION, GOOGLE_API_KEY)
    
    while True:
        query = input("\nWhat now? (or 'quit' or 'forget'): ")
        if query.lower() == 'quit':
            break
        if query.lower() == 'forget':
            history = ''
            continue

        # Query ChromaDB
        results = query_chroma(collection, query, N_RESULTS)
        
        # Prepare context, metadata, and distances from ChromaDB results
        context = "\n\n".join([doc for doc in results['documents'][0]])
        metadata = results['metadatas'][0]
        distances = results['distances'][0]
        
        # Log the full query with context, metadata, and distances
        log_full_query(query, context, metadata, QUERY_FILE)
        
        # Improve question
        print("Improving question with Gemini...")
        improved_question = query_gemini(improve_question_model, context, query)
        
        print(f"\nImproved question: {improved_question}")
        log_conversation(query, improved_question, metadata, distances, CONVERSATION_FILE)

        # Query Gemini
        print("Querying Gemini...")
        query = improved_question
        # Query ChromaDB for new question
        results = query_chroma(collection, query, N_RESULTS)
        # Prepare context, metadata, and distances from ChromaDB results
        context = "\n\n".join([doc for doc in results['documents'][0]])
        metadata = results['metadatas'][0]
        distances = results['distances'][0]
        
        answer = query_gemini(gemini_model, context, query)
        
        # Log the conversation
        log_conversation(query, answer, metadata, distances, CONVERSATION_FILE)

        # Append query and answer to history
        history += '\n' + query + '\n' + answer
        
        # Print Gemini's answer and metadata to the terminal
        print("\nGemini's answer:")
        print(answer)
        print("\nMetadata of sources:")
        for i, (meta, dist) in enumerate(zip(metadata, distances), 1):
            print(f"Source {i}:")
            for key, value in meta.items():
                print(f"  {key}: {value}")
                print(f"  Distance: {dist:.4f}")

if __name__ == "__main__":
    # if len(sys.argv) < 2:
    #     print("Usage: python3 research-main.py <config_file>")
    #     sys.exit(1)
    
    # config_file = sys.argv[1]
    # config = load_config(config_file)
    # main(config)
    main()