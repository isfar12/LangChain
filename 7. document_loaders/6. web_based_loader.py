# playwright install for dynamic web pages
# for static fast web pages, use WebBaseLoader

from langchain_community.document_loaders import PlaywrightURLLoader, WebBaseLoader
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
import time

chat = ChatOllama(
    model="gemma3:1b",
    temperature=0.5
)

prompt = PromptTemplate(
    template="Summarize the product details from the following content: {content}",
    input_variables=["content"]
)

parser = StrOutputParser()


urls=["https://www.applegadgetsbd.com/product/apple-mac-mini-m4-10-core-cpu-10-core-gpu-16-256gb"] # playwright loader format
url = "https://en.wikipedia.org/wiki/Pauline_Ferrand-Pr%C3%A9vot" # web base loader format

loader_web= WebBaseLoader(url)
documents_web = loader_web.load()

loader = PlaywrightURLLoader(urls=urls,headless=True)
documents = loader.load()

print(documents_web)

chain = prompt | chat | parser

begin = time.time()
result = chain.invoke({"content": documents[0].page_content})
end = time.time()

print(result)
print(f"Time taken: {end - begin} seconds")
# Output: Summarized product details from the content loaded from the URL.