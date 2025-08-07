# LangChain Retrievers

This folder contains examples of different types of retrievers available in LangChain. Retrievers are components that fetch relevant documents from a source based on a query.

## 1. Wikipedia Retriever

The `WikipediaRetriever` is a simple retriever that uses the Wikipedia API to fetch documents. It's a great way to quickly get information on a topic.

### Tutorial: `1. wikipedia_retriever.py`

This script demonstrates how to use the `WikipediaRetriever` to fetch the top 2 results for a query about the 1971 war between Bangladesh and Pakistan.

**Code:**

```python
from langchain_community.retrievers import WikipediaRetriever

retriever= WikipediaRetriever(top_k_results=2,lang="en")

query = "1971 War between Bangladesh and Pakistan" 

docs= retriever.invoke(query) # this works like a search engine retrieves the documents related to the query
print(docs)
```

**Explanation:**

1.  **`from langchain_community.retrievers import WikipediaRetriever`**: This line imports the necessary class from the LangChain library.
2.  **`retriever= WikipediaRetriever(top_k_results=2,lang="en")`**: This creates an instance of the `WikipediaRetriever`.
    *   `top_k_results=2`: This parameter specifies that we want to retrieve the top 2 most relevant documents.
    *   `lang="en"`: This sets the language of the Wikipedia to search in to English.
3.  **`query = "1971 War between Bangladesh and Pakistan"`**: This is the query we want to find documents for.
4.  **`docs= retriever.invoke(query)`**: This calls the `invoke` method on the retriever with the query. The retriever then queries the Wikipedia API and returns a list of documents.
5.  **`print(docs)`**: This prints the retrieved documents. Each document will have `page_content` and `metadata` (including the source URL).

## 2. Vector Store Retriever

A `VectorStoreRetriever` uses a vector store to retrieve documents. It embeds the query and finds the most similar documents in the vector store.

### Tutorial: `2. vector_store_retriever.py`

This script shows how to create a `VectorStoreRetriever` from a FAISS vector store.

**Code:**

```python
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings

# Initialize the vector store with documents and embeddings
embeddings = OllamaEmbeddings(model="mxbai-embed-large:latest")


documents = [
    Document(page_content="LangChain helps developers build LLM applications easily."),
    Document(page_content="Chroma is a vector database optimized for LLM-based search."),
    Document(page_content="Embeddings convert text into high-dimensional vectors."),
    Document(page_content="OpenAI provides powerful embedding models."),
]

vector_store = FAISS.from_documents(embedding=embeddings, documents=documents)
vector_store.save_local("faiss_vector_store")
store = FAISS.load_local("faiss_vector_store", OllamaEmbeddings(model="mxbai-embed-large:latest"),allow_dangerous_deserialization=True)

retriever = vector_store.as_retriever(search_kwargs={"k": 2})

results = retriever.invoke("What is LangChain?")

for result in results:
    print(result.page_content)
```

**Explanation:**

1.  **`from langchain_core.documents import Document`**: Imports the `Document` class.
2.  **`from langchain_community.vectorstores import FAISS`**: Imports the `FAISS` vector store.
3.  **`from langchain_ollama import OllamaEmbeddings`**: Imports the embedding model from Ollama.
4.  **`embeddings = OllamaEmbeddings(model="mxbai-embed-large:latest")`**: Initializes the embedding model.
5.  **`documents = [...]`**: Creates a list of `Document` objects.
6.  **`vector_store = FAISS.from_documents(...)`**: Creates a FAISS vector store from the documents and embeddings.
7.  **`vector_store.save_local("faiss_vector_store")`**: Saves the vector store locally.
8.  **`store = FAISS.load_local(...)`**: Loads the vector store from the local directory.
9.  **`retriever = vector_store.as_retriever(search_kwargs={"k": 2})`**: Creates a retriever from the vector store. `search_kwargs={"k": 2}` tells the retriever to return the top 2 most similar documents.
10. **`results = retriever.invoke("What is LangChain?")`**: Invokes the retriever with a query.
11. **`for result in results: print(result.page_content)`**: Prints the content of the retrieved documents.

## 3. Maximal Marginal Relevance (MMR) Retriever

The `Maximal Marginal Relevance (MMR)` retriever is designed to provide diverse results. It selects documents that are relevant to the query but also different from each other.

### Tutorial: `3. mmr_retriever.py`

This script compares the results of a regular retriever with an MMR retriever.

**Code:**

```python
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_community.vectorstores import FAISS


embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")


docs = [
    Document(page_content="LangChain makes it easy to work with LLMs."),
    Document(page_content="LangChain is used to build LLM based applications."),
    Document(page_content="Chroma is used to store and search document embeddings."),
    Document(page_content="Embeddings are vector representations of text."),
    Document(page_content="MMR helps you get diverse results when doing similarity search."),
    Document(page_content="LangChain supports Chroma, FAISS, Pinecone, and more."),
]

vectorstore = FAISS.from_documents(
    documents=docs,
    embedding=embeddings,
)

regular_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

mmr_retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 3, "lambda_mult": 0.5})

query = "What is LangChain?"

regular_results = regular_retriever.invoke(query)
mmr_results = mmr_retriever.invoke(query)

print("Regular Retriever Results:")
for result in regular_results:
    print(result.page_content)
print("\nMMR Retriever Results:")
for result in mmr_results:
    print(result.page_content)
```

**Explanation:**

1.  **`vectorstore = FAISS.from_documents(...)`**: Creates a FAISS vector store with some sample documents.
2.  **`regular_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})`**: Creates a standard vector store retriever that will fetch the top 3 most similar documents.
3.  **`mmr_retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 3, "lambda_mult": 0.5})`**: Creates an MMR retriever.
    *   `search_type="mmr"`: Specifies the search type as Maximal Marginal Relevance.
    *   `search_kwargs={"k": 3, "lambda_mult": 0.5}`:
        *   `k=3`:  The retriever will select 3 documents.
        *   `lambda_mult`: This parameter controls the diversity of the results. A value of `1` gives the most diverse results, while `0` gives the most similar results. `0.5` is a balance between the two.
4.  **`regular_results = regular_retriever.invoke(query)`** and **`mmr_results = mmr_retriever.invoke(query)`**:  The script invokes both retrievers with the same query.
5.  The script then prints the results from both retrievers, allowing you to see the difference in the retrieved documents. The MMR retriever will likely return a more diverse set of documents.

## 4. Multi-Query Retriever

The `MultiQueryRetriever` is a more advanced retriever that uses a language model to generate multiple queries from a user's initial query. It then retrieves documents for each generated query and combines the results. This is useful for complex queries where different aspects might be better captured by different phrasings.

### Tutorial: `4. multi_query_retriever.py`

This script demonstrates how to use the `MultiQueryRetriever`.

**Code:**

```python
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.retrievers.multi_query import MultiQueryRetriever
import logging


# Initialize the vector store with documents and embeddings
embeddings = OllamaEmbeddings(model="mxbai-embed-large")
huggingface_embeddings = HuggingFaceEmbeddings(model_name="intfloat/e5-base-v2")
llm = ChatOllama(model="mistral:7b", temperature=0.1)

# Relevant health & wellness documents
all_docs = [
    # Health-Relevant Documents (First 10)
    Document(page_content="Stretching regularly can increase flexibility and reduce the risk of injury.", metadata={"source": "N1"}),
    Document(page_content="Eating fermented foods like yogurt and kimchi promotes gut health.", metadata={"source": "N2"}),
    Document(page_content="Reading before bed can improve sleep quality and reduce anxiety levels.", metadata={"source": "N3"}),
    Document(page_content="Exposure to natural sunlight helps regulate circadian rhythms and boost vitamin D.", metadata={"source": "N4"}),
    Document(page_content="Hydration supports brain function and helps prevent fatigue throughout the day.", metadata={"source": "N5"}),
    Document(page_content="Regular journaling helps clarify thoughts and improve emotional well-being.", metadata={"source": "N8"}),
    Document(page_content="Listening to music while exercising can enhance endurance and motivation.", metadata={"source": "N9"}),
    Document(page_content="A plant-based diet can lower cholesterol and reduce the risk of heart disease.", metadata={"source": "N11"}),
    Document(page_content="Cold showers may boost circulation and help build mental resilience.", metadata={"source": "N12"}),
    Document(page_content="Social interaction helps lower stress levels and supports mental health.", metadata={"source": "N19"}),

    # Non-Health-Relevant Documents (Last 10)
    Document(page_content="Installing home insulation reduces energy consumption and lowers electricity bills.", metadata={"source": "N6"}),
    Document(page_content="Rust is the result of a chemical reaction between iron, oxygen, and moisture.", metadata={"source": "N7"}),
    Document(page_content="Wireless charging works through electromagnetic induction between coils.", metadata={"source": "N10"}),
    Document(page_content="Learning new skills keeps the brain active and delays cognitive decline.", metadata={"source": "N13"}),  # borderline, but can be seen as cognitive
    Document(page_content="Indoor plants can purify air by absorbing toxins and increasing humidity.", metadata={"source": "N14"}),  # borderline, can affect health indirectly
    Document(page_content="Geothermal energy uses Earth's internal heat to generate clean electricity.", metadata={"source": "N15"}),
    Document(page_content="Moderate caffeine intake can enhance focus but excessive amounts cause jitteriness.", metadata={"source": "N16"}),  # semi-relevant, but kept here
    Document(page_content="Urban green spaces provide psychological relief and encourage physical activity.", metadata={"source": "N17"}),  # borderline, but placed here for balance
    Document(page_content="LED lighting is energy-efficient and has a longer lifespan than traditional bulbs.", metadata={"source": "N18"}),
    Document(page_content="Recycling aluminum saves up to 95% of the energy needed to produce new metal.", metadata={"source": "N20"})
]

vector_store = FAISS.from_documents(embedding=huggingface_embeddings, documents=all_docs)
print(f"Vector store initialized with documents: {vector_store}")


similarity_retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})

multi_query_retriever = MultiQueryRetriever.from_llm(
    llm=llm,
    retriever=vector_store.as_retriever(search_kwargs={"k": 5}),
)
```

**Explanation:**

1.  **`vector_store = FAISS.from_documents(...)`**:  A FAISS vector store is created with a mix of health-related and non-health-related documents.
2.  **`llm = ChatOllama(model="mistral:7b", temperature=0.1)`**: A chat model is initialized, which will be used to generate the different queries.
3.  **`multi_query_retriever = MultiQueryRetriever.from_llm(...)`**: This creates the `MultiQueryRetriever`.
    *   `llm=llm`: The language model to use for generating queries.
    *   `retriever=vector_store.as_retriever(search_kwargs={"k": 5})`: The base retriever to use for fetching documents for each generated query.
4.  When you invoke this retriever, it will first send the original query to the LLM, which will return a list of new queries. Then, it will use the base retriever to find documents for each of those queries, and finally, it will collect and return the unique documents.

## 5. Contextual Compression Retriever

The `ContextualCompressionRetriever` is a powerful tool that helps to improve the quality of retrieved documents by compressing them. It uses a language model to extract only the relevant parts of the documents, thus reducing noise and improving the signal-to-noise ratio.

### Tutorial: `5. context_compress.py`

This script shows how to use the `ContextualCompressionRetriever` with an `LLMChainExtractor`.

**Code:**

```python
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_core.documents import Document
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.retrievers.document_compressors import LLMChainExtractor

embedding_model=HuggingFaceEmbeddings(
    model_name="intfloat/e5-base-v2",
    )

llm = ChatOllama(
    model="mistral:7b",
    temperature=0.5,
)

compressor=LLMChainExtractor.from_llm(llm)



docs = [
    Document(page_content=(
        """The Grand Canyon is one of the most visited natural wonders in the world.
        Photosynthesis is the process by which green plants convert sunlight into energy.
        Millions of tourists travel to see it every year. The rocks date back millions of years."""
    ), metadata={"source": "Doc1"}),

    Document(page_content=(
        """In medieval Europe, castles were built primarily for defense.
        The chlorophyll in plant cells captures sunlight during photosynthesis.
        Knights wore armor made of metal. Siege weapons were often used to breach castle walls."""
    ), metadata={"source": "Doc2"}),

    Document(page_content=(
        """Basketball was invented by Dr. James Naismith in the late 19th century.
        It was originally played with a soccer ball and peach baskets. NBA is now a global league."""
    ), metadata={"source": "Doc3"}),

    Document(page_content=(
        """The history of cinema began in the late 1800s. Silent films were the earliest form.
        Thomas Edison was among the pioneers. Photosynthesis does not occur in animal cells.
        Modern filmmaking involves complex CGI and sound design."""
    ), metadata={"source": "Doc4"})
]

vector_store = FAISS.from_documents(docs, embedding_model)

base_retriever = vector_store.as_retriever(search_kwargs={"k": 5})

compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=base_retriever
)

results = compression_retriever.invoke("What is photosynthesis?")

for result in results:
    print(f"Source: {result.metadata['source']}")
    print(f"Compressed Content: {result.page_content}\n")
```

**Explanation:**

1.  **`llm = ChatOllama(...)`**: Initializes a chat model that will be used for compression.
2.  **`compressor=LLMChainExtractor.from_llm(llm)`**: Creates an `LLMChainExtractor`. This component uses the provided LLM to extract the relevant parts of a document based on a query.
3.  **`vector_store = FAISS.from_documents(...)`**: Creates a FAISS vector store with some sample documents containing mixed information.
4.  **`base_retriever = vector_store.as_retriever(...)`**: Creates a standard retriever.
5.  **`compression_retriever = ContextualCompressionRetriever(...)`**: This creates the `ContextualCompressionRetriever`.
    *   `base_compressor=compressor`: The compressor to use for compressing the documents.
    *   `base_retriever=base_retriever`: The retriever to use for fetching the initial set of documents.
6.  **`results = compression_retriever.invoke("What is photosynthesis?")`**: When this retriever is invoked, it first calls the `base_retriever` to get a set of documents. Then, for each of those documents, it uses the `base_compressor` to extract the parts that are relevant to the query "What is photosynthesis?".
7.  The final output will be a list of compressed documents, containing only the sentences related to photosynthesis.
