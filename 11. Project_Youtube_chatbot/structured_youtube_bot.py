import streamlit as st
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from youtube_transcript_api import YouTubeTranscriptApi
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
import re


# st.write("Hello, YouTube Chatbot!")

chat = ChatOllama(
    model="gemma3:1b",
    temperature=.5
)

embeddings= HuggingFaceEmbeddings(
    model_name="intfloat/e5-base-v2",
    )

# define the function that extracts url id
def extract_youtube_id(url):
    pattern = r"(?:v=|\/)([0-9A-Za-z_-]{11})(?:\?|&|$)"
    match = re.search(pattern, url)
    if match:
        print("Analyzed URL Successfully")
        return match.group(1)
    return None

# define the function that fetches the YouTube transcript and converts into a full text and finally converts into chunks
def youtube_transcript_fetcher(video_id):

    ytt_api = YouTubeTranscriptApi()
    fetched = ytt_api.fetch(video_id)
    raw_data = fetched.to_raw_data()
    full_text = " ".join([item["text"] for item in raw_data])
    print("Transcript Extracted Successfully")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.create_documents([full_text])

    return chunks

# this will format the retrieved documents and convert into text for llm 
def format_docs(docs):
    final_text = "\n\n".join(doc.page_content for doc in docs)
    print("Text Formatted Successfully")
    return final_text


url="https://www.youtube.com/watch?v=DZsz5tVCQBM"


video_id = extract_youtube_id(url)
chunks = youtube_transcript_fetcher(video_id)

# after chunking the text, we create a vector store and create retriever to use them later
vector_store=FAISS.from_documents(chunks, embeddings)
retriever=vector_store.as_retriever(search_type="similarity",search_kwargs={"k":5})

parallel_chain= RunnableParallel(
    {
    # Ensure the retriever receives a string query, not the whole input dict
    'context': retriever | RunnableLambda(format_docs),
    'question': RunnablePassthrough(), # the question will go to the LLM
    }
)

prompt=PromptTemplate(
    template='''
    You are a helpful assistant. Use only the context provided to answer the question. if the answer is not present there, say: "I don't know"
    Context: {context}
    Question: {question}
'''
)
parser = StrOutputParser()
main_chain = parallel_chain | prompt | chat | parser

output = main_chain.invoke("What is the main topic of the video? Write 5 bullet points.")
print(output)