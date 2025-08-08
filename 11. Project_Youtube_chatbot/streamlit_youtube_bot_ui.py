"""
Interactive Streamlit YouTube Chatbot

This app allows users to:
1. Enter a YouTube URL
2. Extract and process the video transcript
3. Ask questions about the video content using RAG (Retrieval Augmented Generation)

The bot uses:
- YouTube Transcript API for transcript extraction
- HuggingFace embeddings for semantic search
- FAISS vector store for document retrieval
- Ollama chat model for question answering
- LangChain for orchestrating the RAG pipeline
"""

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
import time

# Page configuration
st.set_page_config(
    page_title="YouTube Chatbot", 
    page_icon="🎥", 
    layout="wide"
)

st.title("🎥 YouTube Video Chatbot")
st.markdown("Ask questions about any YouTube video's content!")

# Initialize models (cached to avoid reloading)
@st.cache_resource
def initialize_models():
    """Initialize and cache the chat model and embeddings."""
    chat = ChatOllama(
        model="mistral:7b",
        temperature=0.5
    )
    
    embeddings = HuggingFaceEmbeddings(
        model_name="intfloat/e5-base-v2",
    )
    
    return chat, embeddings

def extract_youtube_id(url):
    """Extract YouTube video ID from various URL formats."""
    pattern = r"(?:v=|\/)([0-9A-Za-z_-]{11})(?:\?|&|$)"
    match = re.search(pattern, url)
    if match:
        return match.group(1)
    return None

def youtube_transcript_fetcher(video_id):
    """Fetch YouTube transcript and split into chunks."""
    try:
        # Use the corrected API call
        ytt_api = YouTubeTranscriptApi()
        fetched = ytt_api.fetch(video_id)
        raw_data = fetched.to_raw_data()
        full_text = " ".join([item["text"] for item in raw_data])
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=200
        )
        chunks = text_splitter.create_documents([full_text])
        
        return chunks, full_text
        
    except Exception as e:
        st.error(f"Failed to fetch transcript: {str(e)}")
        return None, None

def format_docs(docs):
    """Format retrieved documents for the LLM context."""
    if not isinstance(docs, (list, tuple)):
        docs = [docs]
    return "\n\n".join(doc.page_content for doc in docs)

def create_rag_chain(retriever, chat_model):
    """Create the RAG chain for question answering."""
    
    # Parallel processing: retrieve context and pass through question
    parallel_chain = RunnableParallel({
        'context': RunnableLambda(lambda x: x["question"]) | retriever | RunnableLambda(format_docs),
        'question': RunnablePassthrough(),
    })
    
    # Prompt template
    prompt = PromptTemplate(
        template='''You are a helpful assistant analyzing a YouTube video transcript. 
Use only the context provided to answer the question. If the answer is not present in the context, say: "I don't know based on the provided transcript."

Context: {context}

Question: {question}

Answer:'''
    )
    
    # Complete chain
    parser = StrOutputParser()
    main_chain = parallel_chain | prompt | chat_model | parser
    
    return main_chain

# Main application logic
def main():
    chat, embeddings = initialize_models()
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        st.markdown("**Model:** Gemma2:1b via Ollama")
        st.markdown("**Embeddings:** intfloat/e5-base-v2")
        st.markdown("**Vector Store:** FAISS")
        st.markdown("**Chunk Size:** 1000 characters")
        
    # URL input
    url_input = st.text_input(
        "🔗 Enter YouTube URL:",
        placeholder="https://www.youtube.com/watch?v=DZsz5tVCQBM",
        help="Paste any YouTube video URL here"
    )
    
    if url_input:
        video_id = extract_youtube_id(url_input)
        
        if not video_id:
            st.error("❌ Invalid YouTube URL. Please check your URL and try again.")
            return
            
        st.success(f"✅ Video ID extracted: `{video_id}`")
        
        # Process video button
        if st.button("🔄 Process Video", type="primary"):
            with st.spinner("Processing video transcript..."):
                
                # Fetch transcript
                chunks, full_text = youtube_transcript_fetcher(video_id)
                
                if chunks is None:
                    return
                
                # Store in session state
                st.session_state.chunks = chunks
                st.session_state.full_text = full_text
                st.session_state.video_processed = True
                
                # Create vector store and retriever
                try:
                    vector_store = FAISS.from_documents(chunks, embeddings)
                    retriever = vector_store.as_retriever(
                        search_type="similarity",
                        search_kwargs={"k": 5}
                    )
                    st.session_state.retriever = retriever
                    
                    # Create RAG chain
                    rag_chain = create_rag_chain(retriever, chat)
                    st.session_state.rag_chain = rag_chain
                    
                    st.success(f"✅ Video processed successfully!")
                    st.info(f"📄 Transcript length: {len(full_text)} characters")
                    st.info(f"📦 Created {len(chunks)} text chunks")
                    
                except Exception as e:
                    st.error(f"Error creating vector store: {str(e)}")
                    return
    
    # Show transcript preview if available
    if hasattr(st.session_state, 'full_text') and st.session_state.full_text:
        with st.expander("📄 View Transcript Preview"):
            st.text_area(
                "First 1000 characters:",
                st.session_state.full_text[:1000] + "...",
                height=200,
                disabled=True
            )
    
    # Question answering section
    if hasattr(st.session_state, 'video_processed') and st.session_state.video_processed:
        st.markdown("---")
        st.header("💬 Ask Questions")
        
        # Predefined sample questions
        st.markdown("**Quick Questions:**")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("What is the main topic?"):
                st.session_state.current_question = "What is the main topic of the video?"
                
        with col2:
            if st.button("Key takeaways"):
                st.session_state.current_question = "What are the key takeaways from this video?"
                
        with col3:
            if st.button("Summary in 5 points"):
                st.session_state.current_question = "Summarize the video in 5 bullet points."
        
        # Custom question input
        question = st.text_input(
            "Or ask your own question:",
            value=getattr(st.session_state, 'current_question', ''),
            placeholder="What specific aspect of the video interests you?"
        )
        
        if question and st.button("🤖 Get Answer", type="secondary"):
            with st.spinner("Thinking..."):
                try:
                    # Get answer using RAG chain
                    result = st.session_state.rag_chain.invoke({
                        "question": question
                    })
                    
                    # Display result
                    st.markdown("### 🎯 Answer:")
                    st.markdown(result)
                    
                    # Store in chat history
                    if 'chat_history' not in st.session_state:
                        st.session_state.chat_history = []
                    
                    st.session_state.chat_history.append({
                        'question': question,
                        'answer': result,
                        'timestamp': time.strftime("%H:%M:%S")
                    })
                    
                except Exception as e:
                    st.error(f"Error generating answer: {str(e)}")
    
    # Chat history
    if hasattr(st.session_state, 'chat_history') and st.session_state.chat_history:
        st.markdown("---")
        st.header("📈 Chat History")
        
        for i, chat in enumerate(reversed(st.session_state.chat_history)):
            with st.expander(f"Q: {chat['question'][:50]}... ({chat['timestamp']})"):
                st.markdown(f"**Question:** {chat['question']}")
                st.markdown(f"**Answer:** {chat['answer']}")

if __name__ == "__main__":
    # Initialize session state
    if 'video_processed' not in st.session_state:
        st.session_state.video_processed = False
        
    main()
