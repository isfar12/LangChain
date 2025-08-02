from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from langchain_ollama import ChatOllama

model = ChatOllama(model="deepscaler", temperature=0.7)

sample_messages = [
    HumanMessage(content="What is the capital of France?"),
    SystemMessage(content="The capital of France is Paris.")
]

result = model.invoke(sample_messages)

sample_messages.append(AIMessage(content=result.content))
print(sample_messages)