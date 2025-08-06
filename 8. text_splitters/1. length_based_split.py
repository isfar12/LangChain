from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader


splitter= CharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=0,
    separator=''
)

pdf_loader=PyPDFLoader(file_path=r"E:\LangChain\7. document_loaders\Documents\Movie Recommend Approach.pdf")

docs=pdf_loader.load()


text= 'Document loaders in LangChain are responsible for loading, parsing, and preparing data from a wide range of sources such as files, websites, APIs, databases, and cloud storage. These loaders convert raw data into a format (typically LangChain `Document` objects) that can be processed for tasks like embedding, chunking, retrieval, and question-answering.\n'

result=splitter.split_text(text)
result_docs=splitter.split_documents(docs)

print("TEXT SPLITTED RESULT:\n\n")
print(result)
print("PDF SPLITTED RESULT:\n\n")
print(result_docs)

'''output
['Document loaders in LangChain are responsible for loading, parsing, and preparing data from a wide r', 'ange of sources such as files, websites, APIs, databases, and cloud storage. These loaders convert r', 'aw data into a format (typically LangChain `Document` objects) that can be processed for tasks like', 'embedding, chunking, retrieval, and question-answering.']
'''