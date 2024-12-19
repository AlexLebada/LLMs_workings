import numpy as np
import llama_index
import csv, json
import pandas as pd
#  pip install llama-index-embeddings-huggingface
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
# pip install google-generativeai
import torch
# pip install llama-index-llms-palm
import math


def read_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return text


def chunk_text(text, chunk_size):
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    return chunks


class LocalEmbeddingModel:
    def __init__(self):
        self.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
        if torch.cuda.is_available() and 1==0:
            self.embed_model.model = self.embed_model.model.to("cuda")
            print("cuda ready")
        else:
            self.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
            print("cuda not ready")

    def get_embedding(self, text: str):
        embedding = self.embed_model.get_text_embedding(text)
        return embedding

    def update_dashboard(self, no_inter:int=1):
        df = pd.read_csv('../LLM_dashboard.csv')
        csv_model = "BAAI/bge-small-en-v1.5"
        df.at[0, csv_model] += no_inter
        df.to_csv('../LLM_dashboard.csv', index=False)



def dot_product(vec1, vec2):
    return sum(a * b for a, b in zip(vec1, vec2))


def magnitude(vec):
    return math.sqrt(sum(v**2 for v in vec))


def cosine_similarity(vec1, vec2):
    dot_prod = dot_product(vec1, vec2)
    mag_vec1 = magnitude(vec1)
    mag_vec2 = magnitude(vec2)

    if mag_vec1 == 0 or mag_vec2 == 0:
        return 0  # Handle division by zero

    return dot_prod / (mag_vec1 * mag_vec2)

def write_to_csv(document_store, csv_file="document_embeddings.csv"):
    documents = document_store.filter_documents(filters=None)


    with open(csv_file, mode="w", newline='', encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Content", "Embedding"])
        for doc in documents:
            embedding_str = json.dumps(doc.embedding)
            writer.writerow([doc.content, embedding_str])

    print(f"Document texts and embeddings written to {csv_file}")