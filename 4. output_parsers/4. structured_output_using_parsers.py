from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate

from langchain.output_parsers import StructuredOutputParser,ResponseSchema

chat =ChatOllama(
    model="gemma:2b",
    temperature=0.5,  # Lower temperature for more consistent JSON output
)

schema=[
    ResponseSchema(name="name", description="The name of the character"),
    ResponseSchema(name="age", description="The age of the character"),
    ResponseSchema(name="description", description="A brief description of the character"),
]

parser=StructuredOutputParser.from_response_schemas(schema)

# The lacking of StructuredOutputParser is that it does not support the validation of the output.
# If you want to validate the output, you can use PydanticOutputParser instead.
template=PromptTemplate(
    template="Write me about of a imaginary character named {name} {format_instructions}",
    input_variables=["name"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)


chain = template | chat | parser

result=chain.invoke({"name": "John Doe"})

print(result)