from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage,SystemMessage


model=ChatOllama(model="llama3.1", temperature=0.7)

while True:
    user=input("You: ")
    if user.lower() in ["exit", "quit"]:
        print("Exiting the chat.")
        break
    llm=model.invoke(user)
    print("LLM:", llm.content)
