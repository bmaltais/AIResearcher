import argparse
import os
from utils import *

QUERY_FILE = 'queries.txt'
CONVERSATION_FILE = './Research.md'

N_RESULTS = 20

SYSTEM_INSTRUCTION = """
You are an experienced research assistant. You will answer with detailed, lengthy university level answers.

Answer the question based on the provided context. If the context doesn't contain enough information to answer the question fully, say so and provide the best answer you can with the available information.

Always provide your answer in markdown.
"""

IMPROVE_QUESTION = """
You are an expert question refiner and context analyzer. Your role is to:
1. Take a user's original question
2. Analyze the provided contextual information
3. Reword and improve the original question based on your own knowledge and with more pertinent words from the provided context

Follow these guidelines:
- Prioritize maintaining the original question's intent and meaning. The improved question should be a natural evolution of the original, not a completely different query.
- Use relevant details from the context to make the question more specific and informative.
- Avoid introducing new information or changing the core topic. The improved question should address the same core concept as the original.
- After the improved question, add instrustions after the question to guide the appropriate style of response for the question.

If the context lacks sufficient information or the original question is already well-formed, simply return the original question.

Always begin your response with your expanded question version and only provide the expanded question version. Do not provide any information about what lead to the improved question.

After the improved question and appropriate instructions, provide a long series of words, concepts taken from the context than are related to the question. Those will help the LLM to retreive related information to the improved question.
"""

N_HISTORY_LINES = 10

def setup_argparse():
    parser = argparse.ArgumentParser(description="ChromaDB and Gemini Query Script")
    parser.add_argument("--collection", type=str, default="default_collection", help="Name of the ChromaDB collection")
    parser.add_argument("--chroma-db-path", type=str, default="./chroma_db", help="Path to store ChromaDB")
    return parser.parse_args()

def main():
    args = setup_argparse()
    
    db = setup_embeddings_and_db(args.chroma_db_path, args.collection)
    gemini_model = setup_gemini('gemini-1.5-pro', SYSTEM_INSTRUCTION, os.getenv('GOOGLE_API_KEY'))
    improve_question_model = setup_gemini('gemini-1.5-pro', IMPROVE_QUESTION, os.getenv('GOOGLE_API_KEY'))
    
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
            print("Improving question with Gemini...")
            improved_question = query_gemini(improve_question_model, context, query)
            
            print(f"\nImproved question: {improved_question}")
            log_conversation(query, improved_question, metadata, distances, CONVERSATION_FILE)

            query = query + "\n\n" + improved_question
            results = query_chromadb(db, query, N_RESULTS)
            context = "\n\n".join([result['documents'] for result in results])
            metadata = [result['metadatas'] for result in results]
            distances = [result['distances'] for result in results]
            
            log_full_query(query, context, metadata, QUERY_FILE)

        print("Querying Gemini...")
        answer = query_gemini(gemini_model, context, query)
        
        log_conversation(query, answer, metadata, distances, CONVERSATION_FILE)

        history += '\n' + query + '\n' + answer
        
        print("\nGemini's answer:")
        print(answer)
        print("\nMetadata of sources:")
        for i, (meta, dist) in enumerate(zip(metadata, distances), 1):
            print(f"Source {i}:")
            for key, value in meta.items():
                print(f"  {key}: {value}")
            print(f"  Distance: {dist:.4f}")

if __name__ == "__main__":
    main()