from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader


docs= DirectoryLoader(
    path=r"E:\LangChain\7. document_loaders\Documents",
    glob="**/*.pdf", # find all the pdf files in the directory , including the subdirectories
    loader_cls=PyPDFLoader, # the method to load the pdf files
)
lists=docs.lazy_load()

for i in lists:
    print(i.metadata) # will load the documents metadata one by one in the memory

lists=docs.load()

for i in lists:
    print(i.metadata) # will load the documents metadata all ( time consuming+ not recommended when there are lot of filess)