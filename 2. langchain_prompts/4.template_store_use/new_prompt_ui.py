from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import PromptTemplate , load_prompt
# This code sets up a simple Streamlit app to interact with an Ollama model
# It allows users to input a message and receive a response from the model

model=ChatOllama(model="phi3", temperature=0.5)


import streamlit as st
# some basic streamlit dropdown options
# these options can be used to customize the prompt dynamically
st.header("Dynamic Prompt Customization")
paper_name = st.selectbox(
    "Select a Paper",
    ["Attention is All You Need", "BERT", "GPT-3", "Diffusion Models", "LLaMA"]
)
explanation_type = st.selectbox(
    "Select Explanation Type",
    ["Beginner Friendly", "Technical Summary", "Key Takeaways"]
)

length = st.selectbox(
    "Select Output Length",
    ["5 lines", "10 lines", "15 lines"]
)
# Now we will just load the prompt template from a file
prompt_template=load_prompt("dynamic_prompt_template.json")

# when the user clicks the "Generate Explanation" button, we will invoke the model with the dynamic prompt
prompt=prompt_template.invoke({
    "paper_name": paper_name,
    "explanation_type": explanation_type,
    "length": length
})

if st.button("Generate Explanation"):
    response = model.invoke(prompt)
    st.write(response.content)