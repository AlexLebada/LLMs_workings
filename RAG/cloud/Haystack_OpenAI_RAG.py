import huggingface_hub
from API_integration.GENERAL_KEYS import HFACE_ENDPOINT_NAME, HFACE_USER_NAME, HFACE_API_KEY
import time, os
import RAG.utility
from haystack.utils import Secret
from haystack import Pipeline
from haystack.components.retrievers import InMemoryEmbeddingRetriever
from haystack.components.builders.prompt_builder import PromptBuilder
from haystack.components.generators import OpenAIGenerator
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.embedders import HuggingFaceAPITextEmbedder
from haystack import Document
import pandas as pd
import ast


huggingface_hub.list_inference_endpoints()
endpoint = huggingface_hub.get_inference_endpoint(
    name=HFACE_ENDPOINT_NAME,  # endpoint name
    namespace=HFACE_USER_NAME,  # user name
)

#endpoint.resume()
endpoint.wait()

start_time = time.time()

csv_file = "./document_embeddings.csv"
data = pd.read_csv(csv_file, encoding='utf-8')
data["Embedding"] = data["Embedding"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)


docstore = InMemoryDocumentStore(embedding_similarity_function="cosine")
documents = [
    Document(content=row["Content"], embedding=row["Embedding"])
    for _, row in data.iterrows()
]
docstore.write_documents(documents)


query = "What a Keeper is supposed to do ?"

template = """
Given the following information, answer the question.

Context: 
{% for document in documents %}
    {{ document.content }}
{% endfor %}

Question: {{ query }}?
"""
pipe = Pipeline()

query_embedder = HuggingFaceAPITextEmbedder(
    api_type="inference_endpoints",
    api_params={"url": "https://gcyju4jiduuxjeex.eu-west-1.aws.endpoints.huggingface.cloud"},
    token=Secret.from_token(HFACE_API_KEY)
)


pipe.add_component("retriever", InMemoryEmbeddingRetriever(document_store=docstore, top_k=5))
pipe.add_component("query_embedder", query_embedder)
pipe.add_component("prompt_builder", PromptBuilder(template=template))
pipe.add_component("llm", OpenAIGenerator(api_key=Secret.from_token(os.environ.get("OPENAI_API_KEY"))))

pipe.connect("query_embedder.embedding", "retriever.query_embedding")
pipe.connect("retriever", "prompt_builder.documents")
pipe.connect("prompt_builder", "llm")


res=pipe.run({
    "query_embedder": {
      "text": query
    },
    "prompt_builder": {
        "query": query
    }
}, include_outputs_from={"retriever", "llm"})

print(res['llm']['replies'])
with open("answer.txt", "w", encoding="utf-8") as file:
    file.write(f"The detailed answer is: \n{res['llm']}\n")
    for idx,doc in enumerate(res["retriever"]["documents"]):
        file.write(f"\nRetriever Document {idx+1}:\n{doc.content}\n")
        file.write("=" * 10 + "\n")

end_time = time.time()
print("Time elapsed:", end_time - start_time)
pipe.draw("rag-pipeline.png")