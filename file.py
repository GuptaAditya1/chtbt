import torch
import ollama
import json
from pdfreader import PDFDocument, SimplePDFViewer
from openai import OpenAI

from pdfreader import SimplePDFViewer, PDFDocument, PageDoesNotExist

def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, "rb") as file:
        # Load the PDF document
        pdf_document = PDFDocument(file)

        # Create a SimplePDFViewer object
        viewer = SimplePDFViewer(file)

        page_number = 1
        while True:
            try:
                viewer.navigate(page_number)
                viewer.render()
                text += " ".join(viewer.canvas.strings) + "\n"
                page_number += 1  # Move to the next page
            except PageDoesNotExist:
                break  # Stop when there are no more pages

    return text

# Example usage
#C:/Users/shale/Downloads/
pdf_path = "C:/Users/shale/Downloads/RESEARCH PAPER.pdf"


pdf_text = extract_text_from_pdf(pdf_path)
#print(pdf_text)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# Example text extracted from the PDF


# Create a Document object
document = Document(page_content=pdf_text)

# Initialize the text splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)

# Split the document into chunks
all_splits = text_splitter.split_documents([document])

# Display the chunks
#for i, split in enumerate(all_splits):
 #   print(f"Chunk {i + 1}:\n{split.page_content}\n")

chunk_size = 500
words = pdf_text.split()
chunks = [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

from transformers import AutoTokenizer, AutoModel

# Specify the path to the locally saved model
local_model_path = "C:/Users/shale/Desktop/local_minilm_v2"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(local_model_path)
model = AutoModel.from_pretrained(local_model_path)

embeddings = []
for chunk in chunks:
    inputs = tokenizer(chunk, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    embeddings.append(outputs.last_hidden_state.mean(dim=1).detach().numpy())
    import faiss
import numpy as np
import re

# Flatten the list of embeddings into a 2D array
embeddings_array = np.vstack(embeddings).astype('float32')

# Build FAISS index
dimension = embeddings_array.shape[1]
faiss_index = faiss.IndexFlatL2(dimension)
faiss_index.add(embeddings_array)
query = "what is rag?"
inputs = tokenizer(query, return_tensors="pt", truncation=True, padding=True)
query_embedding = model(**inputs).last_hidden_state.mean(dim=1).detach().numpy()
distances, indices = faiss_index.search(query_embedding, 5)
retrieved_chunks = [chunks[i] for i in indices[0]]
def retrieve_relevant_chunks(query, index, model, tokenizer, chunks, top_k=5):
    inputs = tokenizer(query, return_tensors="pt", truncation=True, padding=True)
    query_embedding = model(**inputs).last_hidden_state.mean(dim=1).detach().numpy()
    distances, indices = index.search(query_embedding, top_k)
    return [chunks[i] for i in indices[0]]
query = "what is rag?"
retrieved_chunks = retrieve_relevant_chunks(query, faiss_index, model, tokenizer, chunks)
# Import the Ollama client
from ollama import Client

# Initialize the Ollama client
client = Client()

import requests
import time

def generate_response_with_ollama(retrieved_chunks, prompt_template):
    context = " ".join(retrieved_chunks)
    prompt = prompt_template.format(context=context)
    reponse=llm.invoke(input=prompt)
    return response

context = " ".join(retrieved_chunks)
prompt_template = ("""As a personal chat assistant, provide accurate and relevant information based on the provided context in 2-3 sentences. 
        Answe should be limited to 50 words and 2-3 sentences.  
        do not prompt to select answers or do not formualate a stand alone question.
        do not ask questions in the response. Donot give context and donot give model, created at etc 
        
        Context:{context} generate a response to the query:
        """+query )
prompt = prompt_template.format(context=context)

start_time = time.time()

response = client.generate(model="mistral", prompt=prompt)

end_time = time.time()
time_taken = end_time - start_time

## Print the response
#print(response)

#reponse=llm.invoke(input=prompt)

#response = generate_response_with_ollama(retrieved_chunks, prompt_template)
#print(response)
print(time_taken)
response_data=response
def clean_response(response):
    # Remove the reference part (after "Reference:")
    cleaned_response = re.sub(r'Reference:.*', '', response, flags=re.DOTALL)
    
    # Optionally, remove extra whitespace or unwanted parts
    cleaned_response = cleaned_response.strip()

    return cleaned_response

# Cleaned response
cleaned_response = clean_response(response_data['response'])

print(cleaned_response)

