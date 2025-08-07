# Vector Stores in LangChain

## Table of Contents

1. [What are Vector Stores?](#what-are-vector-stores)
2. [What are Vector Databases?](#what-are-vector-databases)
3. [Why Do We Need Vector Stores?](#why-do-we-need-vector-stores)
4. [How Vector Stores Work](#how-vector-stores-work)
5. [Prerequisites](#prerequisites)
6. [Tutorial: Working with Vector Stores](#tutorial-working-with-vector-stores)
7. [Chroma Vector Store Implementation](#chroma-vector-store-implementation)
8. [FAISS Vector Store Implementation](#faiss-vector-store-implementation)
9. [Comparison: Chroma vs FAISS](#comparison-chroma-vs-faiss)
10. [Use Cases and Applications](#use-cases-and-applications)
11. [Best Practices](#best-practices)

## What are Vector Stores?

**Vector stores** are specialized databases designed to store, index, and search high-dimensional vectors efficiently. In the context of AI and machine learning, these vectors are typically **embeddings** - numerical representations of text, images, or other data that capture semantic meaning.

### Key Characteristics:
- **High-dimensional data**: Handle vectors with hundreds to thousands of dimensions
- **Similarity search**: Find vectors that are similar to a query vector
- **Fast retrieval**: Optimized for quick nearest-neighbor searches
- **Scalability**: Handle millions of vectors efficiently

## What are Vector Databases?

**Vector databases** are purpose-built databases for storing and querying vector embeddings. They are the infrastructure layer that powers vector stores.

### Traditional Database vs Vector Database:

| Traditional Database | Vector Database |
|---------------------|-----------------|
| Stores structured data (rows, columns) | Stores high-dimensional vectors |
| Exact matches (WHERE name = 'John') | Similarity searches (find similar vectors) |
| SQL queries | Vector similarity queries |
| Good for transactional data | Good for AI/ML applications |

### Popular Vector Databases:
- **Chroma**: Open-source, easy to use, great for development
- **FAISS**: Facebook's library, very fast, good for production
- **Pinecone**: Cloud-native, managed service
- **Weaviate**: Open-source with GraphQL
- **Qdrant**: High-performance with advanced filtering

## Why Do We Need Vector Stores?

### 1. **Semantic Search**
Traditional keyword search finds exact matches. Vector search finds **meaning**.

**Example**: 
- Query: "Who is the captain?"
- Traditional search: Looks for exact word "captain"
- Vector search: Understands leadership, management, team lead

### 2. **Retrieval Augmented Generation (RAG)**
Vector stores are essential for RAG systems:
1. **Store knowledge**: Convert documents to vectors
2. **Retrieve relevant info**: Find similar content for user queries  
3. **Augment responses**: Provide context to language models

### 3. **Large Language Model Limitations**
- **Context window limits**: Models can't process infinite text
- **Knowledge cutoff**: Models have training data cutoffs
- **Dynamic updates**: Models can't learn new information in real-time

**Vector stores solve this by**:
- Storing unlimited documents
- Providing real-time knowledge updates
- Retrieving only relevant information

### 4. **Efficiency and Scale**
- **Memory efficiency**: Don't load entire datasets into memory
- **Fast retrieval**: Optimized algorithms for similarity search
- **Scalability**: Handle millions of documents

## How Vector Stores Work

### Step-by-Step Process:

1. **Text Input**: "MS Dhoni is the captain of Chennai Super Kings"

2. **Embedding Generation**: 
   ```
   Text → Embedding Model → [0.1, -0.3, 0.8, ..., 0.2]  # 384-dimensional vector
   ```

3. **Storage**: Vector stored with metadata and original text

4. **Query Processing**:
   ```
   Query: "Who leads CSK?" → [0.2, -0.2, 0.7, ..., 0.1]
   ```

5. **Similarity Search**: Find vectors closest to query vector

6. **Results**: Return most similar documents with their text

### Similarity Metrics:
- **Cosine similarity**: Measures angle between vectors
- **Euclidean distance**: Straight-line distance between points
- **Dot product**: Simple multiplication and sum

## Prerequisites

Before running the examples, ensure you have:

### 1. **Python Packages**:
```bash
pip install langchain-ollama langchain-chroma langchain-community faiss-cpu
```

### 2. **Ollama Setup**:
```bash
ollama serve
ollama pull mxbai-embed-large
```

### 3. **Understanding**:
- Basic Python knowledge
- Familiarity with LangChain Document objects
- Understanding of embeddings concept

## Tutorial: Working with Vector Stores

### Sample Data Setup

Our example uses IPL (Indian Premier League) cricket player data:

```python
from langchain.schema import Document

# Creating sample documents with content and metadata
doc1 = Document(
    page_content="Virat Kohli is one of the most successful and consistent batsmen in IPL history. Known for his aggressive batting style and fitness, he has led the Royal Challengers Bangalore in multiple seasons.",
    metadata={"team": "Royal Challengers Bangalore"}
)

doc2 = Document(
    page_content="Rohit Sharma is the most successful captain in IPL history, leading Mumbai Indians to five titles. He's known for his calm demeanor and ability to play big innings under pressure.",
    metadata={"team": "Mumbai Indians"}
)

doc3 = Document(
    page_content="MS Dhoni, famously known as Captain Cool, has led Chennai Super Kings to multiple IPL titles. His finishing skills, wicketkeeping, and leadership are legendary.",
    metadata={"team": "Chennai Super Kings"}
)

doc4 = Document(
    page_content="Jasprit Bumrah is considered one of the best fast bowlers in T20 cricket. Playing for Mumbai Indians, he is known for his yorkers and death-over expertise.",
    metadata={"team": "Mumbai Indians"}
)

doc5 = Document(
    page_content="Ravindra Jadeja is a dynamic all-rounder who contributes with both bat and ball. Representing Chennai Super Kings, his quick fielding and match-winning performances make him a key player.",
    metadata={"team": "Chennai Super Kings"}
)

docs = [doc1, doc2, doc3, doc4, doc5]
```

**Document Structure**:
- **page_content**: The actual text content
- **metadata**: Additional information (team name, categories, etc.)

## Chroma Vector Store Implementation

### 1. **Setup and Configuration**

```python
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

# Initialize Chroma vector store
vector_store = Chroma(
    embedding_function=OllamaEmbeddings(model="mxbai-embed-large:latest"),
    persist_directory="./chroma_vector_store",
    collection_name="ipl_players"
)
```

**Parameters Explained**:
- **embedding_function**: Converts text to vectors using Ollama
- **persist_directory**: Local folder to save the database
- **collection_name**: Name for this specific collection

### 2. **Adding Documents**

```python
# Add all documents to the vector store
vector_store.add_documents(docs)
```

**What happens**:
1. Each document's text gets converted to embeddings
2. Embeddings are stored with metadata and original text
3. Database is persisted to disk

### 3. **Retrieving All Data**

```python
# Get all stored data including embeddings and metadata
all_data = vector_store.get(include=["embeddings", "metadatas", "documents"])
```

**Returns**:
- **documents**: Original text content
- **metadatas**: Associated metadata (team names, etc.)
- **embeddings**: Vector representations
- **ids**: Unique identifiers for each document

### 4. **Similarity Search**

```python
# Search for documents similar to the query
results = vector_store.similarity_search(
    query="Who is the captain of Chennai Super Kings?",
    k=2,  # Return top 2 most similar documents
)
```

**Expected Results**:
```python
[
    Document(
        id='01ee2b2a-27a7-43b9-ba3a-fcf236d546bc',
        metadata={'team': 'Chennai Super Kings'},
        page_content='MS Dhoni, famously known as Captain Cool, has led Chennai Super Kings to multiple IPL titles. His finishing skills, wicketkeeping, and leadership are legendary.'
    ),
    Document(
        id='325b8aab-d371-44a6-94c6-9b4afefe1113',
        metadata={'team': 'Mumbai Indians'},
        page_content="Rohit Sharma is the most successful captain in IPL history, leading Mumbai Indians to five titles. He's known for his calm demeanor and ability to play big innings under pressure."
    )
]
```

**Why these results?**:
1. **MS Dhoni**: Direct match - captain of Chennai Super Kings
2. **Rohit Sharma**: Semantic similarity - also a captain, leadership context

### 5. **Updating Documents**

```python
# Create updated document
updated_doc = Document(
    page_content="Zubayer Isfar is one of the most successful and consistent batsmen in IPL history. Known for his aggressive batting style and fitness, he has led the Royal Challengers Bangalore in multiple seasons.",
    metadata={"team": "Royal Challengers Bangalore"}
)

# Update existing document by ID
vector_store.update_document(
    document_id="8681d921-510f-4a30-be8b-21e2a40e2ba8",
    document=updated_doc,
)
```

**Update Process**:
1. Find document by unique ID
2. Replace content and metadata
3. Regenerate embeddings
4. Update in database

## FAISS Vector Store Implementation

### 1. **Setup and Creation**

```python
from langchain.vectorstores import FAISS

# Create FAISS vector store from documents
faiss_store = FAISS.from_documents(
    documents=docs,
    embedding=OllamaEmbeddings(model="mxbai-embed-large:latest"),
)
```

**Key Differences from Chroma**:
- **No persistence by default**: Stored in memory
- **from_documents()**: Direct creation from document list
- **Faster**: Optimized for speed

### 2. **Saving to Disk**

```python
# Save FAISS index to local directory
faiss_store.save_local("faiss_vector_store_with_meta")
```

**Creates**:
- **index.faiss**: The actual vector index
- **index.pkl**: Metadata and document mappings

### 3. **Loading from Disk**

```python
# Load previously saved FAISS store
store = FAISS.load_local(
    "faiss_vector_store_with_meta", 
    OllamaEmbeddings(model="mxbai-embed-large:latest"),
    allow_dangerous_deserialization=True
)
```

**Security Note**: `allow_dangerous_deserialization=True` is needed for pickle files

### 4. **Performing Searches**

```python
# Search for similar documents
query = "Who is a top finisher?"
results = store.similarity_search(query, k=2)
```

**Expected Results**:
```python
[
    Document(
        metadata={'team': 'Chennai Super Kings'},
        page_content='MS Dhoni, famously known as Captain Cool, has led Chennai Super Kings to multiple IPL titles. His finishing skills, wicketkeeping, and leadership are legendary.'
    ),
    Document(
        metadata={'team': 'Mumbai Indians'},
        page_content="Rohit Sharma is the most successful captain in IPL history, leading Mumbai Indians to five titles. He's known for his calm demeanor and ability to play big innings under pressure."
    )
]
```

**Query Analysis**:
- "top finisher" → semantic similarity with "finishing skills" (MS Dhoni)
- Also matches "successful captain" context

## Comparison: Chroma vs FAISS

### Feature Comparison

| Feature | Chroma | FAISS |
|---------|--------|-------|
| **Ease of Use** | ⭐⭐⭐⭐⭐ Very Easy | ⭐⭐⭐ Moderate |
| **Performance** | ⭐⭐⭐ Good | ⭐⭐⭐⭐⭐ Excellent |
| **Persistence** | ⭐⭐⭐⭐⭐ Built-in | ⭐⭐⭐ Manual |
| **Memory Usage** | ⭐⭐⭐ Moderate | ⭐⭐⭐⭐ Efficient |
| **Scalability** | ⭐⭐⭐⭐ Good | ⭐⭐⭐⭐⭐ Excellent |
| **Metadata Support** | ⭐⭐⭐⭐⭐ Rich | ⭐⭐⭐ Basic |

### When to Use Chroma:
✅ **Development and prototyping**  
✅ **Need rich metadata filtering**  
✅ **Want easy persistence**  
✅ **Building RAG applications**  
✅ **Small to medium datasets**

### When to Use FAISS:
✅ **Production applications**  
✅ **High-performance requirements**  
✅ **Large-scale deployments**  
✅ **Memory-constrained environments**  
✅ **Custom indexing needs**

### Code Comparison

**Chroma - More Declarative**:
```python
# Setup with persistence built-in
vector_store = Chroma(
    embedding_function=embeddings,
    persist_directory="./chroma_db",
    collection_name="my_collection"
)
vector_store.add_documents(docs)
```

**FAISS - More Explicit**:
```python
# Create and manually save
faiss_store = FAISS.from_documents(docs, embeddings)
faiss_store.save_local("faiss_db")

# Later load explicitly
store = FAISS.load_local("faiss_db", embeddings, allow_dangerous_deserialization=True)
```

## Use Cases and Applications

### 1. **Retrieval Augmented Generation (RAG)**

```python
# RAG Pipeline Example
def rag_pipeline(query):
    # 1. Retrieve relevant documents
    relevant_docs = vector_store.similarity_search(query, k=3)
    
    # 2. Create context from retrieved docs
    context = "\n\n".join([doc.page_content for doc in relevant_docs])
    
    # 3. Generate response using context
    prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
    response = llm.invoke(prompt)
    
    return response
```

### 2. **Document Search and Discovery**

```python
# Semantic document search
def find_similar_documents(document_text, top_k=5):
    # Convert query document to embedding
    query_doc = Document(page_content=document_text)
    
    # Find similar documents
    similar_docs = vector_store.similarity_search_by_vector(
        query_doc.page_content, k=top_k
    )
    
    return similar_docs
```

### 3. **Content Recommendation**

```python
# Content recommendation system
def recommend_content(user_preferences, k=3):
    # Create preference query
    pref_text = " ".join(user_preferences)
    
    # Find matching content
    recommendations = vector_store.similarity_search(pref_text, k=k)
    
    return recommendations
```

### 4. **Question Answering Systems**

```python
# QA System
def answer_question(question):
    # Find relevant context
    context_docs = vector_store.similarity_search(question, k=2)
    
    # Extract relevant information
    context = " ".join([doc.page_content for doc in context_docs])
    
    # Generate answer
    answer_prompt = f"""
    Based on the following context, answer the question:
    
    Context: {context}
    Question: {question}
    Answer:
    """
    
    return llm.invoke(answer_prompt)
```

## Best Practices

### 1. **Choosing Chunk Sizes**

```python
# Good chunk sizes for different content types
CHUNK_SIZES = {
    "short_answers": 200,      # Q&A, definitions
    "paragraphs": 500,         # Articles, blogs
    "sections": 1000,          # Documentation
    "pages": 2000,             # Books, reports
}
```

### 2. **Embedding Model Selection**

```python
# Different models for different needs
EMBEDDING_MODELS = {
    "general": "mxbai-embed-large",     # Good all-around
    "code": "text-embedding-ada-002",   # Good for code
    "multilingual": "paraphrase-multilingual-MiniLM-L12-v2"
}
```

### 3. **Metadata Strategy**

```python
# Rich metadata for better filtering
doc = Document(
    page_content="...",
    metadata={
        "source": "wikipedia",
        "category": "sports",
        "subcategory": "cricket", 
        "date": "2024-01-01",
        "author": "sports_writer",
        "language": "en",
        "chunk_id": "chunk_1"
    }
)
```

### 4. **Performance Optimization**

**For Chroma**:
```python
# Batch operations
vector_store.add_documents(docs, batch_size=100)

# Use appropriate distance metrics
vector_store = Chroma(
    embedding_function=embeddings,
    distance_function="cosine"  # or "euclidean", "dot"
)
```

**For FAISS**:
```python
# Use appropriate index types
import faiss

# For exact search (slower but accurate)
index = faiss.IndexFlatIP(dimension)

# For approximate search (faster)
index = faiss.IndexHNSWFlat(dimension, 32)
```

### 5. **Error Handling**

```python
def safe_vector_search(query, k=3):
    try:
        results = vector_store.similarity_search(query, k=k)
        return results
    except Exception as e:
        print(f"Search failed: {e}")
        return []
```

### 6. **Monitoring and Logging**

```python
import time
import logging

def monitored_search(query, k=3):
    start_time = time.time()
    
    try:
        results = vector_store.similarity_search(query, k=k)
        search_time = time.time() - start_time
        
        logging.info(f"Search completed in {search_time:.2f}s, found {len(results)} results")
        return results
        
    except Exception as e:
        logging.error(f"Search failed: {e}")
        return []
```

## Running the Examples

### 1. **Setup Environment**

```bash
# Install required packages
pip install langchain-ollama langchain-chroma langchain-community faiss-cpu

# Start Ollama server
ollama serve

# Download embedding model
ollama pull mxbai-embed-large
```

### 2. **Run the Notebook**

```bash
# Navigate to vector stores directory
cd "e:\LangChain\9. vector_stores"

# Open Jupyter notebook
jupyter notebook "1. chrome_vector.ipynb"
```

### 3. **Expected Directory Structure After Running**

```
9. vector_stores/
├── 1. chrome_vector.ipynb
├── chroma_vector_store/          # Chroma database files
│   ├── chroma.sqlite3
│   └── ...
├── faiss_vector_store_with_meta/ # FAISS index files
│   ├── index.faiss
│   └── index.pkl
└── README.md
```

### 4. **Troubleshooting**

**Common Issues**:

1. **Ollama not running**:
   ```bash
   ollama serve
   ```

2. **Model not found**:
   ```bash
   ollama pull mxbai-embed-large
   ```

3. **FAISS installation issues**:
   ```bash
   # Use CPU version if GPU fails
   pip install faiss-cpu
   ```

4. **Persistence issues**:
   ```python
   # Ensure write permissions in directory
   import os
   os.makedirs("./vector_store", exist_ok=True)
   ```

## Summary

Vector stores are powerful tools that enable semantic search and advanced AI applications:

### Key Benefits:
- **Semantic understanding**: Go beyond keyword matching
- **Scalable storage**: Handle large document collections
- **Fast retrieval**: Optimized for similarity search
- **Flexible metadata**: Rich filtering capabilities

### When to Use:
- **RAG systems**: Provide context to language models
- **Document search**: Find semantically similar content
- **Recommendation engines**: Content discovery
- **Question answering**: Knowledge-based systems

### Getting Started:
1. **Start with Chroma** for development and learning
2. **Use sample data** to understand concepts
3. **Experiment with queries** to see semantic search in action
4. **Move to FAISS** for production performance needs

Vector stores are essential building blocks for modern AI applications, enabling the creation of intelligent, context-aware systems that can understand and retrieve information based on meaning rather than just keywords.
