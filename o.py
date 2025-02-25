import os

# Temporary fix for OpenMP duplicate library issue
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Optional: Limit OpenMP to a single thread
os.environ["OMP_NUM_THREADS"] = "1"

import re
import time
import numpy as np
import faiss
import torch
from flask import Flask, request, jsonify
from pdfreader import PDFDocument, SimplePDFViewer, PageDoesNotExist
from transformers import AutoTokenizer, AutoModel
from ollama import Client

# Flask app initialization
app = Flask(__name__)

# Global variables
pdf_path = "C:/Users/shale/Downloads/RESEARCH PAPER.pdf"  # Hardcoded PDF file path
model_name = 'sentence-transformers/all-MiniLM-L6-v2'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
faiss_index = None
chunks = []
client = Client()

# Function: Extract text from PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, "rb") as file:
        pdf_document = PDFDocument(file)
        viewer = SimplePDFViewer(file)
        page_number = 1
        while True:
            try:
                viewer.navigate(page_number)
                viewer.render()
                text += " ".join(viewer.canvas.strings) + "\n"
                page_number += 1
            except PageDoesNotExist:
                break
    return text

# Function: Split text into chunks
def split_text_into_chunks(text, chunk_size):
    words = text.split()
    return [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

# Function: Generate embeddings using a local model
def generate_embeddings(chunks, model, tokenizer):
    embeddings = []
    for chunk in chunks:
        inputs = tokenizer(chunk, return_tensors="pt", truncation=True, padding=True)
        outputs = model(**inputs)
        embeddings.append(outputs.last_hidden_state.mean(dim=1).detach().numpy())
    return np.vstack(embeddings).astype('float32')

# Function: Retrieve relevant chunks
def retrieve_relevant_chunks(query, index, model, tokenizer, chunks, top_k=5):
    inputs = tokenizer(query, return_tensors="pt", truncation=True, padding=True)
    query_embedding = model(**inputs).last_hidden_state.mean(dim=1).detach().numpy()
    distances, indices = index.search(query_embedding, top_k)
    return [chunks[i] for i in indices[0]]

# Function: Generate response using the Ollama client
def generate_response(retrieved_chunks, query, client, model_name="mistral"):
    context = " ".join(retrieved_chunks)
    prompt_template = (
        f"As a personal chat assistant, provide accurate and relevant information based on the provided context in 2-3 sentences. "
        f"Answer should be limited to 50 words and 2-3 sentences. "
        f"Do not include context, model, or any metadata in the response. \n\nContext:{context}\n\nQuery: {query}"
    )
    response = client.generate(model=model_name, prompt=prompt_template)
    return response

# Function: Clean response
def clean_response(response):
    cleaned_response = re.sub(r'Reference:.*', '', response, flags=re.DOTALL).strip()
    return cleaned_response

# Process the PDF during app initialization
def process_pdf():
    global faiss_index, chunks
    try:
        # Extract and process the PDF
        pdf_text = extract_text_from_pdf(pdf_path)
        chunks = split_text_into_chunks(pdf_text, chunk_size=500)
        embeddings = generate_embeddings(chunks, model, tokenizer)

        # Create FAISS index
        dimension = embeddings.shape[1]
        faiss_index = faiss.IndexFlatL2(dimension)
        faiss_index.add(embeddings)

        print("PDF processed and indexed successfully!")
    except Exception as e:
        print(f"Error processing PDF: {str(e)}")

# Endpoint: Query
@app.route('/query', methods=['POST'])
def query_pdf():
    global faiss_index, chunks
    if faiss_index is None or not chunks:
        return jsonify({"error": "No PDF processed. Please check the configuration."}), 500
    
    data = request.json
    query = data.get('query', '').strip()
    if not query:
        return jsonify({"error": "Query cannot be empty."}), 400

    try:
        retrieved_chunks = retrieve_relevant_chunks(query, faiss_index, model, tokenizer, chunks)
        response_data = generate_response(retrieved_chunks, query, client)
        cleaned_response = clean_response(response_data['response'])
        return jsonify({"response": cleaned_response}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Initialize the PDF processing
process_pdf()

# Run Flask app
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
