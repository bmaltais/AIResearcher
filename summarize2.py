from langchain_community.document_loaders import PyPDFLoader
from langchain_community.llms import Ollama
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.schema import Document

# Taking out the warnings
import warnings
from warnings import simplefilter

# Filter out FutureWarnings
simplefilter(action='ignore', category=FutureWarning)

llm = Ollama(model="gemma2", temperature=0)

def extract_text_from_txt(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()

# Load the book
text = extract_text_from_txt('H:\open-webui\data\docs\cloud\pg2641.txt')
# pages = loader.load()

# Combine the pages, and replace the tabs with spaces
# text = "".join(page.page_content for page in pages).replace('\t', ' ')

num_tokens = llm.get_num_tokens(text)
print(f"This book has {num_tokens} tokens in it")

text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", "\t"], chunk_size=2000, chunk_overlap=600)
docs = text_splitter.create_documents([text])
num_documents = len(docs)
print(f"Now our book is split up into {num_documents} documents")

embeddings = OllamaEmbeddings(model="gemma2") # nomic-embed-text")
vectors = embeddings.embed_documents([x.page_content for x in docs])

# Convert the list of embeddings to a numpy array
vectors_array = np.array(vectors)

# Choose the number of clusters
num_clusters = 11

# Perform K-means clustering
kmeans = KMeans(n_clusters=num_clusters, random_state=42).fit(vectors_array)
print(kmeans.labels_)

# Perform t-SNE and reduce to 2 dimensions
tsne = TSNE(n_components=2, random_state=42)
reduced_data_tsne = tsne.fit_transform(vectors_array)

# Plot the reduced data
plt.figure(figsize=(10, 8))
scatter = plt.scatter(reduced_data_tsne[:, 0], reduced_data_tsne[:, 1], c=kmeans.labels_, cmap='viridis')
plt.colorbar(scatter)
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.title('Book Embeddings Clustered')
plt.show()

# Find the closest embeddings to the centroids

# Create an empty list that will hold your closest points
closest_indices = []

# Loop through the number of clusters you have
for i in range(num_clusters):
    
    # Get the list of distances from that particular cluster center
    distances = np.linalg.norm(vectors - kmeans.cluster_centers_[i], axis=1)
    
    # Find the list position of the closest one (using argmin to find the smallest distance)
    closest_index = np.argmin(distances)
    
    # Append that position to your closest indices list
    closest_indices.append(closest_index)
    
selected_indices = sorted(closest_indices)

print(selected_indices)

llm3 = ChatOpenAI(temperature=0, model_name="gpt-4o-mini", max_tokens=1000)

map_prompt = """
You will be given text enclosed in triple backticks (```)
Your goal is to condense the text as if you were the author of this text and asked to reduce it to 1000 words. Use the same style and structure as the provided passage.

```{text}```
FULL SUMMARY:
"""
map_prompt_template = PromptTemplate(template=map_prompt, input_variables=["text"])

map_chain = load_summarize_chain(llm=llm3,
                             chain_type="stuff",
                             prompt=map_prompt_template)

selected_docs = [docs[doc] for doc in selected_indices]

# Make an empty list to hold your summaries
summary_list = []

# Loop through a range of the lenght of your selected docs
for i, doc in enumerate(selected_docs):
    
    # Go get a summary of the chunk
    chunk_summary = map_chain.run([doc])
    
    # Append that summary to your list
    summary_list.append(chunk_summary)
    
    print (f"Summary #{i} (chunk #{selected_indices[i]}) - Preview: {chunk_summary[:250]} \n")
    
summaries = "\n".join(summary_list)

# Convert it back to a document
summaries = Document(page_content=summaries)

print (f"Your total summary has {llm.get_num_tokens(summaries.page_content)} tokens")

llm4 = ChatOpenAI(temperature=0, model_name="gpt-4o-mini", max_tokens=3000)

combine_prompt = """
You will be given a series of text abstracts. The text abstracts will be enclosed in triple backticks (```)
Your goal is to use the provided text and write a meaningfull and cohesive output from it as if you wrote the text.

```{text}```
VERBOSE SUMMARY:
"""
combine_prompt_template = PromptTemplate(template=combine_prompt, input_variables=["text"])

reduce_chain = load_summarize_chain(llm=llm4,
                             chain_type="stuff",
                             prompt=combine_prompt_template,
                             verbose=True # Set this to true if you want to see the inner workings
                                   )

output = reduce_chain.run([summaries])

print (output)