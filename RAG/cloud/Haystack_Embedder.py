from haystack import Document
from haystack import Pipeline
import RAG.utility
from haystack.utils import Secret
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.embedders import HuggingFaceAPITextEmbedder, HuggingFaceAPIDocumentEmbedder
from haystack.components.writers import DocumentWriter
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from API_integration.GENERAL_KEYS import HFACE_API_KEY
import time


start_time = time.time()
document_store = InMemoryDocumentStore(embedding_similarity_function="cosine")
text_file = RAG.utility.read_text_file("../texts/Harry_Potter_1.txt")
chunks = RAG.utility.chunk_text(text_file, chunk_size=250)
documents = []

for chunk in chunks:
    documents.append(Document(content=chunk))

print(len(documents))

document_embedder = HuggingFaceAPIDocumentEmbedder(
    api_type="inference_endpoints",
    api_params={"url": "https://gcyju4jiduuxjeex.eu-west-1.aws.endpoints.huggingface.cloud"},
    token=Secret.from_token(HFACE_API_KEY)
)


indexing_pipeline = Pipeline()
indexing_pipeline.add_component("document_embedder", document_embedder)
indexing_pipeline.add_component("doc_writer", DocumentWriter(document_store=document_store))
indexing_pipeline.connect("document_embedder", "doc_writer")
indexing_pipeline.run({"document_embedder": {"documents": documents}})

RAG.utility.write_to_csv(document_store)


end_time = time.time()

print("elapsed:", end_time - start_time)

# Visualise pipeline
indexing_pipeline.draw("index-pipeline.png")