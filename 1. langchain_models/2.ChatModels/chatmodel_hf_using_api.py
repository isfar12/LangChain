from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from google import genai

from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

LLM = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1",
    task="text-generation",
)



# The client gets the API key from the environment variable `GEMINI_API_KEY`.
client = genai.Client()

response = client.models.generate_content(
    model="gemini-2.5-flash", contents="Explain how AI works in a few words"
)
print(response.text)

# model=ChatHuggingFace(llm=LLM)

# result=model.invoke("What is the capital of Bangladesh?")
# print(result.content)
