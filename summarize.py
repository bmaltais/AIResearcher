import os
from dotenv import load_dotenv
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import ChatOpenAI
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain_core.prompts import PromptTemplate

# Option 2

from langchain.chains import MapReduceDocumentsChain, ReduceDocumentsChain
from langchain_text_splitters import CharacterTextSplitter

# Option 3

from langchain_community.llms import Ollama
from langchain.schema import Document
from typing import List

load_dotenv()

LANGCHAIN_TRACING_V2 = os.getenv('LANGCHAIN_TRACING_V2')
LANGCHAIN_API_KEY = os.getenv('LANGCHAIN_API_KEY')

loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
docs = loader.load()
# llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini")
llm = Ollama(model="mistral", temperature=0)

# # Define prompt
# prompt_template = """Write a concise summary of the following:
# "{text}"
# CONCISE SUMMARY:"""
# prompt = PromptTemplate.from_template(prompt_template)

# # Define LLM chain

# llm_chain = LLMChain(llm=llm, prompt=prompt)

# # Define StuffDocumentsChain
# stuff_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="text")

# docs = loader.load()
# print(stuff_chain.invoke(docs)["output_text"])

# Option 2

# llm = ChatOpenAI(temperature=0)

# # Map
# map_template = """The following is a set of documents
# {docs}
# Based on this list of docs, please identify the main themes 
# Helpful Answer:"""
# map_prompt = PromptTemplate.from_template(map_template)
# map_chain = LLMChain(llm=llm, prompt=map_prompt)

# # Reduce
# reduce_template = """The following is set of summaries:
# {docs}
# Take these and distill it into a final, consolidated summary of the main themes. 
# Helpful Answer:"""
# reduce_prompt = PromptTemplate.from_template(reduce_template)

# # Run chain
# reduce_chain = LLMChain(llm=llm, prompt=reduce_prompt)

# # Takes a list of documents, combines them into a single string, and passes this to an LLMChain
# combine_documents_chain = StuffDocumentsChain(
#     llm_chain=reduce_chain, document_variable_name="docs"
# )

# # Combines and iteratively reduces the mapped documents
# reduce_documents_chain = ReduceDocumentsChain(
#     # This is final chain that is called.
#     combine_documents_chain=combine_documents_chain,
#     # If documents exceed context for `StuffDocumentsChain`
#     collapse_documents_chain=combine_documents_chain,
#     # The maximum number of tokens to group documents into.
#     token_max=4000,
# )

# # Combining documents by mapping a chain over them, then combining results
# map_reduce_chain = MapReduceDocumentsChain(
#     # Map chain
#     llm_chain=map_chain,
#     # Reduce chain
#     reduce_documents_chain=reduce_documents_chain,
#     # The variable name in the llm_chain to put the documents in
#     document_variable_name="docs",
#     # Return the results of the map steps in the output
#     return_intermediate_steps=False,
# )

# text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
#     chunk_size=1000, chunk_overlap=0
# )
# split_docs = text_splitter.split_documents(docs)

# result = map_reduce_chain.invoke(split_docs)

# print(result["output_text"])

# Option 3

text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=1024, chunk_overlap=0
)
split_docs = text_splitter.split_documents(docs)
prompt_template = """
Write a concise summary of the following as if you were the author of the document. Keep the same style at the original document:

{text}

CONCISE SUMMARY:
"""
prompt = PromptTemplate.from_template(prompt_template)

refine_template = """
Your job is to produce a final summary as if you were the author of the document.
Keep the same writing style as the document.
We have provided an existing summary up to a certain point:
------------
{existing_answer}
------------
We have the opportunity to refine the existing summary (only if needed) with some more context below.
------------
{text}
------------
Given the new context, refine the original summary in English. If the context isn't useful, return the original summary.
"""
refine_prompt = PromptTemplate.from_template(refine_template)
chain = load_summarize_chain(
    llm=llm,
    chain_type="refine",
    question_prompt=prompt,
    refine_prompt=refine_prompt,
    return_intermediate_steps=True,
    input_key="input_documents",
    output_key="output_text",
)
result = chain.invoke({"input_documents": split_docs}, return_only_outputs=True)

print("Text summary:\n\n")
print(result["output_text"])

# print("Intermediate steps:\n\n")
# print("\n\n".join(result["intermediate_steps"][:3]))