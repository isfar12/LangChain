from langchain_community.retrievers import WikipediaRetriever

retriever= WikipediaRetriever(top_k_results=2,lang="en")

query = "1971 War between Bangladesh and Pakistan" 

docs= retriever.invoke(query) # this works like a search engine retrieves the documents related to the query
print(docs)