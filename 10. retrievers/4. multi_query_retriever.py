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

# Example query to retrieve relevant documents
query = "How can I help reduce the risk of health issues?"

print(f"Original Query: {query}\n")

similarity_results = similarity_retriever.invoke(query)

multi_query_results = multi_query_retriever.invoke(query)


print("Similarity Retriever Results:")
for i, doc in enumerate(similarity_results, 1):
    print(f"{i}. {doc.page_content} (Source: {doc.metadata['source']})")

print("\n" + "="*50 + "\n")

print("Multi-Query Retriever Results:")
for i, doc in enumerate(multi_query_results, 1):
    print(f"{i}. {doc.page_content} (Source: {doc.metadata['source']})")
