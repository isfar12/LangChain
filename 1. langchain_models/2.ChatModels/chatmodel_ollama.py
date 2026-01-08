from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage,SystemMessage


# using downloaded local ollama model

chat = ChatOllama(model="gemma3:1b", temperature=0.7)

messages=[
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="What is the capital of France?")
]
response = chat.invoke(messages)
print(response.content)