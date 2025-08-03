from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

chat_prompt = ChatPromptTemplate(
    [
        ("system", "You are a helpful assistant."),
        MessagesPlaceholder(variable_name="chat_history"), # this will be replaced with the chat history    
        ("human", "{user_input}")
    ]
)
chat_history = []

with open("2. langchain_prompts/8.dynamic_chat_history/chat_history.txt", "r") as file:
    chat_history.extend(file.readlines())

user_input = "When will I get my refund?"
result = chat_prompt.invoke({
    "chat_history": chat_history,
    "user_input": user_input
})
print(result)
