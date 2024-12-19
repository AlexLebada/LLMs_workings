from haystack.components.embedders import HuggingFaceAPIDocumentEmbedder
from haystack.utils import Secret
from haystack.dataclasses import Document
from API_integration.GENERAL_KEYS import HFACE_API_KEY

doc = Document(content="My name is alex")

document_embedder = HuggingFaceAPIDocumentEmbedder(
    api_type="inference_endpoints",
    api_params={"url": "https://gcyju4jiduuxjeex.eu-west-1.aws.endpoints.huggingface.cloud"},
    token=Secret.from_token(HFACE_API_KEY)
)

result = document_embedder.run([doc])