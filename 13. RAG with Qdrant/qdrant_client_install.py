from qdrant_client import QdrantClient, models
from dotenv import load_dotenv
import os

QDRANT_URL =os.getenv("QDRANT_URL", "https://qdrant-ai-1-0-0-1-0-0-1-0-0-1-0-0.a.run.app")
QDRANT_API_KEY =os.getenv("QDRANT_API_KEY", "")
client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)