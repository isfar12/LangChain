from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity

embeddings = HuggingFaceEmbeddings(
    model_name="./models/sentence-transformers/all-MiniLM-L6-v2"
)

documents=[
    "The capital of Bangladesh is Dhaka.",
    "The capital of France is Paris.",
    "The capital of Japan is Tokyo."
]

query = "What is the capital of Bangladesh?"

doc_embeddings = embeddings.embed_documents(documents)
query_embedding = embeddings.embed_query(query)

print(cosine_similarity([query_embedding],doc_embeddings))
