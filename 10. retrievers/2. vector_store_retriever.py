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