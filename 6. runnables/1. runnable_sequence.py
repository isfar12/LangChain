from langchain_core.runnables import RunnableSequence
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

model= ChatOllama(
    model="gemma3:1b",
    temperature=0.7
)

prompt1 = PromptTemplate(
    template="Write me a joke about {topic}",
    input_variables=["topic"]
)

prompt2 = PromptTemplate(
    template="Explain the following joke: {joke} \n First write the joke, then explain it.",
    input_variables=["joke"]
)

parser = StrOutputParser()

# Create a sequence of runnables by chaining them together sequentially one after another
chain = RunnableSequence( prompt1, model, parser, prompt2, model, parser)

print(chain.invoke({"topic": "cats"}))