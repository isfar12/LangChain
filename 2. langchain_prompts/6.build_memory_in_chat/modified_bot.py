from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage,SystemMessage,AIMessage


model=ChatOllama(model="deepscaler", temperature=0.7)

simple_history = [
    SystemMessage(content="You are a helpful assistant.") # storing the system message (for context)
]


while True:
    user=input("You: ")
    simple_history.append(HumanMessage(content=user))
    if user.lower() in ["exit", "quit"]:
        print("Exiting the chat.")
        break
    result=model.invoke(simple_history)
    simple_history.append(AIMessage(content=result.content))
    llm=result.content

    print("LLM:", llm)
