from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
import streamlit as st

chat = ChatOllama(model="phi3", temperature=0.7)

st.header("Chat with Ollama Model")

user_input = st.text_input("Enter your message: ")

if st.button("Send"):
    if user_input:
        messages=[
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content=user_input)
        ]
        response = chat.invoke(messages)
        st.write(f"Response: {response.content}")