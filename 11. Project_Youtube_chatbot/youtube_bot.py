import re
from youtube_transcript_api import YouTubeTranscriptApi
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate


chat = ChatOllama(
    model="mistral:7b",
    temperature=.5
)


embeddings= HuggingFaceEmbeddings(
    model_name="intfloat/e5-base-v2",
    )

def extract_youtube_id(url):
    pattern = r"(?:v=|\/)([0-9A-Za-z_-]{11})(?:\?|&|$)"
    match = re.search(pattern, url)
    if match:
        return match.group(1)
    return None


video_id = extract_youtube_id("https://www.youtube.com/watch?v=tqPQB5sleHY")  # Example YouTube video ID
print(f"Extracted Video ID: {video_id}")

ytt_api = YouTubeTranscriptApi()
fetched = ytt_api.fetch(video_id)
raw_data = fetched.to_raw_data()

full_text = " ".join([item["text"] for item in raw_data])
print("Text Extracted Successfully")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.create_documents([full_text])

vector_store=FAISS.from_documents(chunks, embeddings)
print("Vector Store Created Successfully")

retriever=vector_store.as_retriever(search_type="similarity",search_kwargs={"k":5})

question="What is the topic about?"
result = retriever.invoke(question)
print("Retrieval Successfully")

context ="\n\n".join(doc.page_content for doc in result)


prompt=PromptTemplate(
    template='''
    You are a helpful assistant. Use only the context provided to answer the question. if the answer is not present there, say: "I don't know"
    Context: {context}
    Question: {question}
'''
)

final_prompt=prompt.format(context=context, question=question)

chat=ChatOllama(
    model="mistral:7b",
    temperature=.5
)

answer=chat.invoke(final_prompt)
print("Answer Generated Successfully\n")
print(answer)
