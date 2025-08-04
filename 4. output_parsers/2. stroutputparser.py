from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
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
# string output parser to parse the model's response so that it can be used in the next prompt easily
#instead of using invoke method, we can use the chain method to chain the prompts and model together
parser = StrOutputParser()


chain = template1 | model | parser | template2 | model | parser

result=chain.invoke({"topic": "Artificial Intelligence"})
print(result)