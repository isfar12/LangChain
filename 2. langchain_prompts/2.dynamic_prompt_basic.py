from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import PromptTemplate
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
# The user can select a paper, explanation type, and output length
# These selections will be used to generate a dynamic prompt
# After selecting options, we will use a custom prompt template to generate a dynamic prompt

template = """
Please provide a detailed explanation of the following paper:

Paper Name: {paper_name}
Explanation Type: {explanation_type}
Output Length: {length}
1. If mathematical equations are present, explain them in simple terms.
2. Provide a beginner-friendly overview of the paper.
3. Include a technical summary for advanced readers.
4. Use related examples to illustrate key concepts.

Ensure the explanation is clear and concise, tailored to the selected output length.

"""
# Create a PromptTemplate instance with the dynamic template
prompt_template = PromptTemplate(
    template=template,
    input_variables=["paper_name", "explanation_type", "length"]
)
# when the user clicks the "Generate Explanation" button, we will invoke the model with the dynamic prompt
prompt=prompt_template.invoke({
    "paper_name": paper_name,
    "explanation_type": explanation_type,
    "length": length
})

if st.button("Generate Explanation"):
    response = model.invoke(prompt)
    st.write(response.content)