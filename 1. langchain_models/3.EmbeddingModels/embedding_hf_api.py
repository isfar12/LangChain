import os
HUGGINGFACEHUB_API_TOKEN=os.getenv("HUGGINGFACEHUB_API_TOKEN")

from langchain_huggingface import HuggingFaceEndpointEmbeddings
model = "sentence-transformers/all-mpnet-base-v2"
hf = HuggingFaceEndpointEmbeddings(
    model=model,
    task="feature-extraction",
    huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
)

result=hf.embed_query("hello world")
result2=hf.embed_documents["hello world","bye bye"]