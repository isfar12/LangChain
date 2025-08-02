from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

LLM=HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1",
    task="text-generation",
)

model=ChatHuggingFace(llm=LLM)

result=model.invoke("What is the capital of Bangladesh?")
print(result.content)