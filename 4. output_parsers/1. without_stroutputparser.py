from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama


model=ChatOllama(
    model="gemma:2b",
    temperature=0.5,  # Lower temperature for more consistent output
)
# This code demonstrates how to use the ChatOllama model with multiple prompts to generate a report and then summarize it by the same model.
template1=PromptTemplate(
    template="Write me a detailed report on {topic}",
    input_variables=["topic"],
)

template2=PromptTemplate(
    template="Summarize the key points of {topic}",
    input_variables=["topic"],
)

# this is the step by step process to generate a report and then summarize it
# using the same model with different prompts.

prompt1=template1.invoke({"topic": "Artificial Intelligence"})

result=model.invoke(prompt1)

prompt2=template2.invoke({"topic":result.content})

result2=model.invoke(prompt2)
print(result.content)
print(result2.content)