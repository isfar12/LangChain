from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="./models/sentence-transformers/all-MiniLM-L6-v2"
)
 
# Example usage
result = embeddings.embed_query("What is the capital of Bangladesh?")
print(result)

"""
That line:

No sentence-transformers model found with name ./models/sentence-transformers/all-MiniLM-L6-v2. Creating a new one with mean pooling.
is NOT an error — it's a notice coming from the sentence-transformers library when you load a model using langchain_huggingface.HuggingFaceEmbeddings.


"""
