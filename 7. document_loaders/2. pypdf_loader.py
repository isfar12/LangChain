from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import time

document_path = r"E:\LangChain\7. document_loaders\Documents\Movie Recommend Approach.pdf"

loader = PyPDFLoader(document_path)
document = loader.load()
print(len(document))

print(document[0].page_content) 
