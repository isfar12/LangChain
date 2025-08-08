from youtube_transcript_api import YouTubeTranscriptApi
from langchain.text_splitter import RecursiveCharacterTextSplitter

video_id = "T-D1OfcDW1M"  # Example YouTube video ID

ytt_api = YouTubeTranscriptApi()
fetched = ytt_api.fetch(video_id)
raw_data = fetched.to_raw_data()

full_text = " ".join([item["text"] for item in raw_data])
print(full_text)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.create_documents([full_text])
print(len(chunks))
