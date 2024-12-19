#from utility import chunk_text, get_embedding, read_text_file,cosine_similarity
from RAG import utility
from API_integration.GENERAL_KEYS import MONGO_DB_KEY
from pymongo import MongoClient
from API_integration.GENERAL_KEYS import PALM_API_KEY
import google.generativeai as genai
import time
import numpy as np
from joblib import Parallel, delayed

palm_api_key = PALM_API_KEY
genai.configure(api_key=palm_api_key)

cluster = MongoClient(MONGO_DB_KEY)
db = cluster["LLM_work"]
collection = db["embeddings_RAG"]


embd_model = utility.LocalEmbeddingModel()
start_time = time.time()
text_file = utility.read_text_file("../texts/Harry_Potter_1.txt")
chunks = utility.chunk_text(text_file, chunk_size=250)
vdb = []


vdb = Parallel(n_jobs=-1)(delayed(embd_model.get_embedding)(text) for text in chunks[:30])

end_time = time.time()

print("elapsed:", end_time - start_time)

records = [{"text": text, "embeddings": embedding} for text, embedding in zip(chunks, vdb)]
collection.insert_many(records)

query = "what is greatest fear of dursleys"
q_embd = embd_model.get_embedding(query)
ratings = [utility.cosine_similarity(q_embd, x) for x in vdb]
k = 1
idx = np.argpartition(ratings, -k)[-k:]



prompt = f"You are a smart agent. A question would be asked to you and relevant information would be provided.\
    Your task is to answer the question and use the information provided. Question - {query}. Relevant Information - {[chunks[index] for index in idx]}"


model = genai.GenerativeModel("gemini-1.5-flash")

# prompt request
try:
    response = model.generate_content(prompt)
    print("API response:", response.text)
except Exception as e:
    print("Error:", e)

