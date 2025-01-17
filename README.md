* ## Semantic_search
      Model embedding:text-embedding-ada-002

I built a semantic search script using OpenAI API, that takes some reviews with dense data, extract only texts  which is transformed in vector embeddings. 
These text-embedding pairs are stored in MongoDB Atlas and then returns first top_k texts by highest cosine similarity scores.
Dataset used are reviews from: https://raw.githubusercontent.com/keitazoumana/Experimentation-Data/main/Musical_instruments_reviews.csv

### Evaluation 
For now I checked the results and when I input a short positive sentiment("I like this" or " I dont like it") there are some bad returns, but when using longer text with certain word similarity, 
that are existing in the stored text, there is a visible difference in their similarity value. 
<br/>
<br/>
* ## RAG
RAG is a technique to get LLM answers based on context provided by external data. For this I used a text file 'Harry_Potter_1.txt', having a total size of 400 000 chars from Harry Potter story.
Without any preprocessing and split in chunks of 250chars, I've tried different types of deployment and models, to make some incremental improvement

### Local/RAG_1.py
      Local model embedding: BAAI/bge-small-en-v1.5
      Model for answer generation: Gemini API/gemini-1.5-flash
      Hardware: CPU - Ryzen7: 3.2Ghz, 8core, 16threads ; 24GB Ram,  GPU - Geforce gtx1650 
    
When tested a sequential computation it took me something like 52s for 30 chunks, and 13mins for entire text to get the relevant answer.
For parallel processing, the time is half, but still too much. I tried to swap to GPU with cuda, but for some reason I couldnt initialize the GPU usage

### Cloud
    Deployment platform: HuggingFace
    Framework to build pipeline : Haystack 2.8
    Cloud hardware: CPU - Intel Sapphire Rapids · 4x vCPUs · 8 GB
    Embedding model: BAAI/bge-base-en-v1.5
    Model for answer generation: OpenAI/gpt-4o

As I wanted to get relevant answers for big documents, I had to make cloud deployment for vector embedding computation.
<br/>

<ins>Haystack_Embedder.py:</ins>
     Is where it takes a .txt, split it in chunks and returns as a .csv with chunk-embedding pairs, by the pipeline described in index-pipeline.png.
     It takes ~130s for the entire text(1756 chunks) to be processed, which is pretty nice for my goal
    
![Screenshot 2024-12-20 155834](https://github.com/user-attachments/assets/42125173-9dde-4914-9230-9e5795cda016)

<ins>Haystack_OpenAI_RAG.py:</ins>
     takes user query and returns relevant chunks from .csv by semantic search. Query and retrieved texts are passed to llm as a predefined prompt template.
     The answer in answer.txt is fast and mostly accurate

![Screenshot 2024-12-20 161857](https://github.com/user-attachments/assets/a4651bd5-35a1-401f-a5d5-4754042d6f77)

