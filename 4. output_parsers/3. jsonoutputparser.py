from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

chat =ChatOllama(
    model="gemma:2b",
    temperature=0.5,  # Lower temperature for more consistent JSON output
)

parser=JsonOutputParser()


template=PromptTemplate(
    template="Write me about of a imaginary character {name} {format_instructions}",
    input_variables=["name"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)


chain = template | chat | parser

result=chain.invoke({"name": "John Doe"})

print(result) 
# the main problem with jsonoutputparser is that it does not support structured output
# so we will use StructuredOutputParser instead in the next example
# and we will use ResponseSchema to define the structure of the output