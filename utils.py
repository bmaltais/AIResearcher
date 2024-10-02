import os
from datetime import datetime
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

load_dotenv()

def setup_embeddings_and_db(chroma_db_path, collection_name="default_collection"):
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

def setup_gemini(model_name, system_instruction, api_key):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name=model_name, safety_settings=None, system_instruction=system_instruction)    
    return model

def query_gemini(model, context, question):
    prompt = f"""Context: \n{context}\n\nQuestion: \n{question}"""
    response = model.generate_content(prompt)
    return response.text

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

def improve_question(improve_question_model, db, query, n_results):
    results = query_chromadb(db, query, n_results)
    context = "\n\n".join([result['documents'] for result in results])
    print("Improving question with Gemini...")
    improved_question = query_gemini(improve_question_model, context, query)
    print(f"\nImproved question: {improved_question}")
    return improved_question, results

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