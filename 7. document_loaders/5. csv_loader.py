from langchain_community.document_loaders import CSVLoader
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate


chat = ChatOllama(
    model="gemma3:1b",
    temperature=0.5
)

parser= StrOutputParser()

prompt = PromptTemplate(
    template="Extract the first 5 rows from the CSV file: {file} \n\nand explain what the data represents.",
    input_variables=["file"],
)

document_loader = CSVLoader(
    file_path=r"E:\LangChain\7. document_loaders\Documents\Student Mental health.csv",
)
documents = document_loader.load()

chain = prompt | chat | parser

# result=chain.invoke({"file": documents[:5]})
first_5 = "\n\n".join([doc.page_content for doc in documents[:5]])

print(first_5)
print(chain.invoke({"file": first_5}))
# The result will contain the first 5 rows of the CSV file along with an explanation of the data.
# Make sure to adjust the file path as necessary for your environment.