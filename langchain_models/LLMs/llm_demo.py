#using downloaded local ollama model

from langchain_ollama import OllamaLLM

llm=OllamaLLM(model="phi3", temperature=0.7)

print(llm.invoke("What is the capital of France?"))
