from haystack.components.embedders import HuggingFaceAPIDocumentEmbedder
from haystack.utils import Secret
from haystack.dataclasses import Document
from API_integration.GENERAL_KEYS import HFACE_API_KEY

doc = Document(content="I love pizza!")

document_embedder = HuggingFaceAPIDocumentEmbedder(
    api_type="serverless_inference_api",
    api_params={"model": "BAAI/bge-small-en-v1.5"},
    token=Secret.from_token(HFACE_API_KEY)
)

result = document_embedder.run([doc])
print(len(result["documents"][0].embedding))