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
