# HuggingFace Local Embeddings with LangChain

## Overview
This document explains the `embedding_hf_local.py` code that demonstrates how to use locally downloaded HuggingFace embedding models with LangChain.

## Code Breakdown

### Import Statement
```python
from langchain_huggingface import HuggingFaceEmbeddings
```
- Imports the `HuggingFaceEmbeddings` class from LangChain's HuggingFace integration
- This class provides a wrapper around HuggingFace embedding models for use with LangChain

### Model Initialization
```python
embeddings = HuggingFaceEmbeddings(
    model_name="./models/sentence-transformers/all-MiniLM-L6-v2"
)
```

**Key Points:**
- **Local Path**: Uses `"./models/sentence-transformers/all-MiniLM-L6-v2"` to point to the locally downloaded model
- **Model Choice**: `all-MiniLM-L6-v2` is a popular sentence transformer model that:
  - Produces 384-dimensional embeddings
  - Good balance between performance and speed
  - Suitable for semantic similarity tasks
- **Parameter**: `model_name` accepts both local paths and HuggingFace Hub model names

### Text Embedding
```python
result = embeddings.embed_query("What is the capital of Bangladesh?")
print(len(result))
```

**Explanation:**
- **`embed_query()`**: Method to convert a single text string into a vector embedding
- **Input**: A text question "What is the capital of Bangladesh?"
- **Output**: A numerical vector (list of floats) representing the semantic meaning of the text
- **`len(result)`**: Prints the dimensionality of the embedding vector (should be 384 for this model)

## What are Embeddings?

Embeddings are numerical representations of text that capture semantic meaning:
- **Vector Representation**: Text is converted to a list of numbers (vector)
- **Semantic Similarity**: Similar texts have similar vectors
- **Dimensionality**: Each vector has a fixed number of dimensions (384 for this model)
- **Use Cases**: Search, clustering, similarity comparison, recommendation systems

## Model Details: all-MiniLM-L6-v2

- **Type**: Sentence Transformer
- **Dimensions**: 384
- **Max Sequence Length**: 256 tokens
- **Performance**: Good for general-purpose sentence embeddings
- **Size**: Relatively small (~90MB)
- **Speed**: Fast inference

## Advantages of Local Models

1. **No Internet Required**: Works offline once downloaded
2. **Privacy**: Data doesn't leave your machine
3. **Speed**: No network latency
4. **Cost**: No API fees
5. **Consistency**: Same model version always

## Common Use Cases

### 1. Semantic Search
```python
query_embedding = embeddings.embed_query("search term")
document_embeddings = embeddings.embed_documents(["doc1", "doc2", "doc3"])
# Compare similarities using cosine similarity
```

### 2. Text Clustering
```python
texts = ["text1", "text2", "text3"]
text_embeddings = embeddings.embed_documents(texts)
# Use clustering algorithms on embeddings
```

### 3. Similarity Comparison
```python
text1_embedding = embeddings.embed_query("First text")
text2_embedding = embeddings.embed_query("Second text")
# Calculate cosine similarity between vectors
```

### 4. Document Similarity Analysis
```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Sample documents
documents = [
    "Machine learning is a subset of artificial intelligence.",
    "Deep learning uses neural networks with multiple layers.",
    "Python is a popular programming language for data science.",
    "Natural language processing helps computers understand human language.",
    "AI and ML are transforming various industries today."
]

# Get embeddings for all documents
doc_embeddings = embeddings.embed_documents(documents)

# Convert to numpy array for easier computation
embeddings_array = np.array(doc_embeddings)

# Calculate similarity matrix
similarity_matrix = cosine_similarity(embeddings_array)

# Find most similar documents
def find_most_similar(query_text, documents, top_k=3):
    query_embedding = embeddings.embed_query(query_text)
    doc_embeddings = embeddings.embed_documents(documents)
    
    # Calculate similarities
    similarities = cosine_similarity([query_embedding], doc_embeddings)[0]
    
    # Get top-k most similar documents
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    results = []
    for idx in top_indices:
        results.append({
            'document': documents[idx],
            'similarity': similarities[idx],
            'index': idx
        })
    
    return results

# Example usage
query = "What is artificial intelligence?"
similar_docs = find_most_similar(query, documents)

for i, result in enumerate(similar_docs):
    print(f"{i+1}. Similarity: {result['similarity']:.4f}")
    print(f"   Document: {result['document']}")
    print()
```

## Alternative Methods

### Multiple Documents
```python
# For multiple texts at once
documents = ["Text 1", "Text 2", "Text 3"]
doc_embeddings = embeddings.embed_documents(documents)
```

### With Custom Parameters
```python
embeddings = HuggingFaceEmbeddings(
    model_name="./models/sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'},  # or 'cuda' for GPU
    encode_kwargs={'normalize_embeddings': True}
)
```

## Expected Output

When you run the code, you should see:
```
384
```

This indicates that each text is converted into a 384-dimensional vector.

## Troubleshooting

### Common Issues:
1. **Model Not Found**: Ensure the model is downloaded in the correct path
2. **Memory Issues**: Large models may require more RAM
3. **Slow Performance**: Consider using GPU if available

### Model Download:
If you need to download the model:
```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
model.save('./models/sentence-transformers/all-MiniLM-L6-v2')
```

## Integration with LangChain

This embedding model can be used with various LangChain components:
- **Vector Stores**: Chroma, FAISS, Pinecone
- **Retrievers**: For RAG (Retrieval Augmented Generation)
- **Document Search**: Semantic document retrieval
- **Question Answering**: Vector-based Q&A systems

## Document Similarity Applications

### 1. Content Recommendation System
```python
def recommend_similar_content(target_doc, document_pool, top_k=5):
    """
    Recommend similar documents based on semantic similarity
    """
    target_embedding = embeddings.embed_query(target_doc)
    pool_embeddings = embeddings.embed_documents(document_pool)
    
    similarities = cosine_similarity([target_embedding], pool_embeddings)[0]
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    recommendations = []
    for idx in top_indices:
        recommendations.append({
            'content': document_pool[idx],
            'similarity_score': similarities[idx]
        })
    
    return recommendations
```

### 2. Duplicate Detection
```python
def find_duplicates(documents, threshold=0.9):
    """
    Find potential duplicate documents based on similarity threshold
    """
    doc_embeddings = embeddings.embed_documents(documents)
    similarity_matrix = cosine_similarity(doc_embeddings)
    
    duplicates = []
    for i in range(len(documents)):
        for j in range(i+1, len(documents)):
            if similarity_matrix[i][j] > threshold:
                duplicates.append({
                    'doc1_index': i,
                    'doc2_index': j,
                    'doc1': documents[i][:100] + "...",
                    'doc2': documents[j][:100] + "...",
                    'similarity': similarity_matrix[i][j]
                })
    
    return duplicates
```

### 3. Document Clustering
```python
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def cluster_documents(documents, n_clusters=3):
    """
    Cluster documents based on semantic similarity
    """
    doc_embeddings = embeddings.embed_documents(documents)
    
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(doc_embeddings)
    
    # Group documents by cluster
    clusters = {}
    for i, label in enumerate(cluster_labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(documents[i])
    
    return clusters, cluster_labels

# Example usage
documents = [
    "Python programming tutorial",
    "Machine learning algorithms",
    "Web development with JavaScript",
    "Data science with Python",
    "React.js frontend development",
    "Deep learning neural networks"
]

clusters, labels = cluster_documents(documents, n_clusters=2)
for cluster_id, docs in clusters.items():
    print(f"Cluster {cluster_id}:")
    for doc in docs:
        print(f"  - {doc}")
    print()
```

### 4. Semantic Search Engine
```python
class SemanticSearchEngine:
    def __init__(self, documents):
        self.documents = documents
        self.embeddings_model = embeddings
        self.doc_embeddings = self.embeddings_model.embed_documents(documents)
    
    def search(self, query, top_k=5):
        query_embedding = self.embeddings_model.embed_query(query)
        similarities = cosine_similarity([query_embedding], self.doc_embeddings)[0]
        
        # Get top results
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append({
                'document': self.documents[idx],
                'score': similarities[idx],
                'rank': len(results) + 1
            })
        
        return results
    
    def add_document(self, new_doc):
        self.documents.append(new_doc)
        new_embedding = self.embeddings_model.embed_documents([new_doc])
        self.doc_embeddings.extend(new_embedding)

# Example usage
search_engine = SemanticSearchEngine([
    "Introduction to machine learning concepts",
    "Python programming for beginners",
    "Advanced neural network architectures",
    "Web scraping techniques with Python",
    "Data visualization using matplotlib"
])

search_results = search_engine.search("learning algorithms")
for result in search_results:
    print(f"Rank {result['rank']}: {result['document']}")
    print(f"Score: {result['score']:.4f}\n")
```

## Summary

The `embedding_hf_local.py` demonstrates:
1. Loading a local HuggingFace embedding model
2. Converting text to numerical vectors
3. Using LangChain's HuggingFace integration
4. Working with semantic representations of text

This setup is ideal for applications requiring text understanding, similarity comparison, and semantic search capabilities while maintaining privacy and offline functionality.