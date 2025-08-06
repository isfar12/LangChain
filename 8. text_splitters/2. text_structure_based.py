from langchain.text_splitter import CharacterTextSplitter,RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader


splitter= RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=0,
)

text= '''Document loaders in LangChain are responsible for loading, parsing, and preparing data from a wide range of sources such as files, websites, APIs, databases, and cloud storage. These loaders convert raw data into a format (typically LangChain `Document` objects) that can be processed for tasks like embedding, chunking, retrieval, and question-answering.

The movieId, title, and genres columns will be useful. You can treat the genres column as the items features (as a list of genres). By transforming this text data into numerical form (e.g., using one-hot encoding or TF-IDF), you can calculate cosine similarity between items based on their genres. The movieId will connect it with the ratings data frame.
'''

print("TEXT SPLITTED RESULT:\n\n")
result=splitter.split_text(text)
print(result)

'''result

['Document loaders in LangChain are responsible for loading, parsing, and preparing data from a wide range of sources such as files, websites, APIs, databases, and cloud storage. These loaders convert', 'raw data into a format (typically LangChain `Document` objects) that can be processed for tasks like embedding, chunking, retrieval, and question-answering.', 'The movieId, title, and genres columns will be useful. You can treat the genres column as the items features (as a list of genres). By transforming this text data into numerical form (e.g., using', 'one-hot encoding or TF-IDF), you can calculate cosine similarity between items based on their genres. The movieId will connect it with the ratings data frame.']
'''