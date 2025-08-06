from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader


docs= DirectoryLoader(
    path=r"E:\LangChain\7. document_loaders\Documents",
    glob="**/*.pdf", # find all the pdf files in the directory , including the subdirectories
    loader_cls=PyPDFLoader, # the method to load the pdf files
)
lists=docs.load()

print(len(lists))
print(lists[20].page_content) 