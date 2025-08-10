# LangChain Retrievers

## Overview
This folder contains examples of different retriever types in LangChain. Retrievers fetch relevant documents from data sources based on a query.

## Table of Contents
1. [Wikipedia Retriever](#1-wikipedia-retriever)
2. [Vector Store Retriever](#2-vector-store-retriever)
3. [MMR Retriever](#3-mmr-retriever)
4. [Multi-Query Retriever](#4-multi-query-retriever)
5. [Contextual Compression Retriever](#5-contextual-compression-retriever)

## 1. Wikipedia Retriever

Uses Wikipedia's API to fetch real-time information from Wikipedia articles.

### Features:
- Direct access to Wikipedia's knowledge base
- Multi-language support
- Automatic document formatting

### Tutorial: `1. wikipedia_retriever.py`

**Step 1: Import and Setup**
```python
from langchain_community.retrievers import WikipediaRetriever
```

**Step 2: Create Retriever**
```python
retriever = WikipediaRetriever(top_k_results=2, lang="en")
```

**Step 3: Define Query and Search**
```python
query = "1971 War between Bangladesh and Pakistan"
docs = retriever.invoke(query)
print(docs)
```

**Key Parameters:**
- `top_k_results`: Number of articles to retrieve
- `lang`: Language code (en, es, fr, etc.)

**Use Cases:** Historical research, fact-checking, educational content

## 2. Vector Store Retriever

Uses vector embeddings for semantic similarity search in document collections.

### Features:
- Semantic understanding (meaning-based search)
- Fast similarity calculations with FAISS
- Persistent storage and loading
- Configurable similarity thresholds

### Tutorial: `2. vector_store_retriever.py`

**Step 1: Import Required Libraries**
```python
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
```

**Step 2: Initialize Embedding Model**
```python
embeddings = OllamaEmbeddings(model="mxbai-embed-large:latest")
```

**Step 3: Create Documents**
```python
documents = [
    Document(page_content="LangChain helps developers build LLM applications easily."),
    Document(page_content="Chroma is a vector database optimized for LLM-based search."),
    Document(page_content="Embeddings convert text into high-dimensional vectors."),
    Document(page_content="OpenAI provides powerful embedding models."),
]
```

**Step 4: Build Vector Store**
```python
vector_store = FAISS.from_documents(embedding=embeddings, documents=documents)
vector_store.save_local("faiss_vector_store")
```

**Step 5: Create Retriever and Search**
```python
retriever = vector_store.as_retriever(search_kwargs={"k": 2})
results = retriever.invoke("What is LangChain?")

for result in results:
    print(result.page_content)
```

**Key Parameters:**
- `k`: Number of similar documents to return
- `search_type`: "similarity" or "similarity_score_threshold"
- `score_threshold`: Minimum similarity score for results

**Use Cases:** Enterprise search, document Q&A, content recommendation
## 3. MMR Retriever

Maximal Marginal Relevance (MMR) balances relevance and diversity in search results to avoid redundant documents.

### Features:
- Reduces result redundancy
- Configurable relevance vs diversity balance
- Better topic coverage

### Tutorial: `3. mmr_retriever.py`

**Step 1: Import Libraries and Setup**
```python
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")
```

**Step 2: Create Test Documents**
```python
docs = [
    Document(page_content="LangChain makes it easy to work with LLMs."),
    Document(page_content="LangChain is used to build LLM based applications."),
    Document(page_content="Chroma is used to store and search document embeddings."),
    Document(page_content="Embeddings are vector representations of text."),
    Document(page_content="MMR helps you get diverse results when doing similarity search."),
    Document(page_content="LangChain supports Chroma, FAISS, Pinecone, and more."),
]
```

**Step 3: Build Vector Store**
```python
vectorstore = FAISS.from_documents(documents=docs, embedding=embeddings)
```

**Step 4: Create Both Retriever Types**
```python
regular_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
mmr_retriever = vectorstore.as_retriever(
    search_type="mmr", 
    search_kwargs={"k": 3, "lambda_mult": 0.5}
)
```

**Step 5: Compare Results**
```python
query = "What is LangChain?"

regular_results = regular_retriever.invoke(query)
mmr_results = mmr_retriever.invoke(query)

print("Regular Results:")
for result in regular_results:
    print(result.page_content)

print("\nMMR Results:")
for result in mmr_results:
    print(result.page_content)
```

**Key Parameters:**
- `lambda_mult`: Controls relevance vs diversity (0.0 = max diversity, 1.0 = max relevance)
- `fetch_k`: Number of documents to fetch before MMR filtering

**Use Cases:** Research exploration, recommendation systems, content curation

## 4. Multi-Query Retriever

Uses an LLM to generate multiple variations of your query, searches with each, and combines results for comprehensive coverage.

### Features:
- Automatic query expansion using LLM
- Covers different phrasings and perspectives
- Reduces query sensitivity
- Better recall for complex questions

### Tutorial: `4. multi_query_retriever.py`

**Step 1: Import Libraries**
```python
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.retrievers.multi_query import MultiQueryRetriever
```

**Step 2: Initialize Models**
```python
embeddings = OllamaEmbeddings(model="mxbai-embed-large")
huggingface_embeddings = HuggingFaceEmbeddings(model_name="intfloat/e5-base-v2")
llm = ChatOllama(model="mistral:7b", temperature=0.1)
```

**Step 3: Create Document Collection**
```python
all_docs = [
    Document(page_content="Stretching regularly can increase flexibility and reduce the risk of injury.", metadata={"source": "N1"}),
    Document(page_content="Eating fermented foods like yogurt and kimchi promotes gut health.", metadata={"source": "N2"}),
    Document(page_content="Reading before bed can improve sleep quality and reduce anxiety levels.", metadata={"source": "N3"}),
    Document(page_content="Exposure to natural sunlight helps regulate circadian rhythms and boost vitamin D.", metadata={"source": "N4"}),
    Document(page_content="Hydration supports brain function and helps prevent fatigue throughout the day.", metadata={"source": "N5"}),
]
```

**Step 4: Build Vector Store**
```python
vector_store = FAISS.from_documents(embedding=huggingface_embeddings, documents=all_docs)
```

**Step 5: Create Retrievers**
```python
similarity_retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})
multi_query_retriever = MultiQueryRetriever.from_llm(
    llm=llm,
    retriever=vector_store.as_retriever(search_kwargs={"k": 5}),
)
```

**Step 6: Compare Results**
```python
query = "How can I help reduce the risk of health issues?"
similarity_results = similarity_retriever.invoke(query)
multi_query_results = multi_query_retriever.invoke(query)

print("Similarity Results:")
for i, doc in enumerate(similarity_results, 1):
    print(f"{i}. {doc.page_content}")

print("\nMulti-Query Results:")
for i, doc in enumerate(multi_query_results, 1):
    print(f"{i}. {doc.page_content}")
```

**Key Parameters:**
- `llm`: Language model for generating query variations
- `retriever`: Base retriever to search with each generated query
- `prompt`: Custom prompt template for query generation

**Use Cases:** Research, complex Q&A, cross-domain queries

## 5. Contextual Compression Retriever

Combines retrieval with LLM-based extraction to return only the relevant parts of documents, reducing noise and improving precision.

### Features:
- LLM-powered content extraction
- Reduces information overload
- Filters out irrelevant sections
- Maintains document metadata

### Tutorial: `5. context_compress.py`

**Step 1: Import Libraries**
```python
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_core.documents import Document
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.retrievers.document_compressors import LLMChainExtractor
```

**Step 2: Initialize Models**
```python
embedding_model = HuggingFaceEmbeddings(model_name="intfloat/e5-base-v2")
llm = ChatOllama(model="mistral:7b", temperature=0.5)
compressor = LLMChainExtractor.from_llm(llm)
```

**Step 3: Create Mixed-Content Documents**
```python
docs = [
    Document(page_content="""The Grand Canyon is one of the most visited natural wonders in the world.
    Photosynthesis is the process by which green plants convert sunlight into energy.
    Millions of tourists travel to see it every year.""", metadata={"source": "Doc1"}),
    
    Document(page_content="""In medieval Europe, castles were built primarily for defense.
    The chlorophyll in plant cells captures sunlight during photosynthesis.
    Knights wore armor made of metal.""", metadata={"source": "Doc2"}),
    
    Document(page_content="""Basketball was invented by Dr. James Naismith in the late 19th century.
    It was originally played with a soccer ball and peach baskets.""", metadata={"source": "Doc3"}),
    
    Document(page_content="""The history of cinema began in the late 1800s.
    Photosynthesis does not occur in animal cells.
    Modern filmmaking involves complex CGI.""", metadata={"source": "Doc4"})
]
```

**Step 4: Build Vector Store and Retrievers**
```python
vector_store = FAISS.from_documents(docs, embedding_model)
base_retriever = vector_store.as_retriever(search_kwargs={"k": 5})

compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=base_retriever
)
```

**Step 5: Query and Get Compressed Results**
```python
query = "What is photosynthesis?"
results = compression_retriever.invoke(query)

for result in results:
    print(f"Source: {result.metadata['source']}")
    print(f"Compressed Content: {result.page_content}\n")
```

**Expected Output:**
```
Source: Doc1
Compressed Content: Photosynthesis is the process by which green plants convert sunlight into energy.

Source: Doc2
Compressed Content: The chlorophyll in plant cells captures sunlight during photosynthesis.

Source: Doc4
Compressed Content: Photosynthesis does not occur in animal cells.
```

**Key Parameters:**
- `base_compressor`: LLM-based extractor for relevant content
- `base_retriever`: Standard retriever for initial document selection
- `temperature`: LLM creativity level for extraction decisions

**Use Cases:** Long document processing, precision-focused search, information extraction
