import os
import tempfile
from typing import List, Dict, Any
from uuid import uuid4
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_groq import ChatGroq
from langchain_qdrant import Qdrant
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from qdrant_client import QdrantClient, models as qmodels

load_dotenv()

HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

if not all([HUGGINGFACEHUB_API_TOKEN, GROQ_API_KEY, QDRANT_URL, QDRANT_API_KEY]):
    missing = [k for k,v in {
        "HUGGINGFACEHUB_API_TOKEN": HUGGINGFACEHUB_API_TOKEN,
        "GROQ_API_KEY": GROQ_API_KEY,
        "QDRANT_URL": QDRANT_URL,
        "QDRANT_API_KEY": QDRANT_API_KEY
    }.items() if not v]
    raise RuntimeError(f"Missing required environment variables: {missing}")


COLLECTION_NAME = "documents"
EMBED_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
EMBED_DIM = 768  
app = FastAPI(title="RAG with FastAPI + Qdrant", version="1.0.0")


embeddings = HuggingFaceEndpointEmbeddings(
    model=EMBED_MODEL_NAME,
    task="feature-extraction",
    huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
)

chat = ChatGroq(model="openai/gpt-oss-20b", temperature=0)

qclient = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

if not qclient.collection_exists(collection_name=COLLECTION_NAME):
    qclient.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=qmodels.VectorParams(
            size=EMBED_DIM, distance=qmodels.Distance.COSINE
        )
    )


vectorstore = Qdrant(
    client=qclient,
    collection_name=COLLECTION_NAME,
    embeddings=embeddings,
)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800, chunk_overlap=120, separators=["\n\n", "\n", " ", ""]
)

def _save_to_tmp(upload: UploadFile) -> str:
    suffix = os.path.splitext(upload.filename or "file.pdf")[-1] or ".pdf"
    fd, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "wb") as f:
        f.write(upload.file.read())
    return path

def _load_pdf_as_docs(path: str, source_name: str) -> List[Any]:
    loader = PyPDFLoader(path)
    docs = loader.load()

    for d in docs:
        d.metadata = d.metadata or {}
        d.metadata["source"] = source_name
    return docs

def _chunk_docs(docs: List[Any]) -> List[Any]:
    return text_splitter.split_documents(docs)

def _format_for_prompt(docs: List[Any]) -> str:
    lines = []
    for i, d in enumerate(docs, start=1):
        page = d.metadata.get("page", "")
        src = d.metadata.get("source", "")
        lines.append(f"[{i}] (p.{page}) {d.page_content.strip()}\nSOURCE: {src}")
    return "\n\n".join(lines[:10])




class IngestResult(BaseModel):
    total_chunks: int
    chunks_per_file: Dict[str, int]

class QueryRequest(BaseModel):
    question: str
    top_k: int = 4

class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict[str, str]]




@app.get("/")
def root():
    return {"status": "ok", "service": "RAG with FastAPI + Qdrant"}

@app.post("/ingest", response_model=IngestResult)
async def ingest(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")

    chunks_per_file = {}
    total_chunks = 0

    for upload in files:
        if not (upload.filename or "").lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail=f"Only PDF accepted: {upload.filename}")

        tmp_path = _save_to_tmp(upload)
        try:
            pages = _load_pdf_as_docs(tmp_path, source_name=upload.filename)
            chunks = _chunk_docs(pages)
            doc_id = str(uuid4())
            for c in chunks:
                c.metadata["doc_id"] = doc_id

            vectorstore.add_documents(chunks)
            chunks_per_file[upload.filename] = len(chunks)
            total_chunks += len(chunks)
        finally:
            try:
                os.remove(tmp_path)
            except Exception:
                pass

    return IngestResult(total_chunks=total_chunks, chunks_per_file=chunks_per_file)

@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    retriever = vectorstore.as_retriever(search_kwargs={"k": req.top_k})
    docs = retriever.get_relevant_documents(req.question)
    context = _format_for_prompt(docs)

    system = (
        "You are a careful assistant. Answer the user's question using ONLY the context. "
        "If the answer isn't in the context, say you don't know. Keep the answer concise."
    )
    prompt = (
        f"<SYSTEM>\n{system}\n</SYSTEM>\n\n"
        f"<CONTEXT>\n{context}\n</CONTEXT>\n\n"
        f"<QUESTION>\n{req.question}\n</QUESTION>"
    )

    result = chat.invoke(prompt)
    answer = (getattr(result, "content", None) or str(result)).strip()

    sources = []
    for d in docs:
        sources.append({
            "source": d.metadata.get("source", "unknown"),
            "page": str(d.metadata.get("page", "")),
        })

    return QueryResponse(answer=answer, sources=sources)
