#using downloaded local ollama model

from langchain_ollama import OllamaLLM

llm=OllamaLLM(model="gemma3:270m", temperature=0.7)

print(llm.invoke("What is the capital of France?"))
