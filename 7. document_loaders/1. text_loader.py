from langchain_community.document_loaders import TextLoader
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import time

model= ChatOllama(
    model="gemma2:2b",
    temperature=0.5,
)

prompt = PromptTemplate(
    template= "Write the summary of the following document: {text}",
    input_variables=["text"],
)

parser =StrOutputParser()

loader= TextLoader(r"E:\LangChain\7. document_loaders\Documents\Langchain DocumentLoader Types.txt", encoding="utf-8")

document = loader.load()

chain = prompt | model | parser

start = time.time()
result = chain.invoke({"text": document[0].page_content})

print(result)

end = time.time()
print(f"Processing time: {end - start} seconds")

print(len(document))
print(type(document))
print(type(document[0]))
print(document[0])