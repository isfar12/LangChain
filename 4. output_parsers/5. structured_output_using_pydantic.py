from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import PydanticOutputParser
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from dotenv import load_dotenv
load_dotenv()

chat = ChatOllama(
    model="phi3:latest",  # More reliable model for structured output
    temperature=0.1,      # Lower temperature for more consistent output
)


model_google = ChatGoogleGenerativeAI(
    model="gemini-3-flash-preview",
    temperature=1.0,  # Gemini 3.0+ defaults to 1.0
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)

# the format we want our output to be in


class Person(BaseModel):
    name: str = Field(description="The name of the character")
    age: int = Field(gt=20, description="The age of the character")
    description: str = Field(
        description="A brief description of the character")


# using pydanticoutputparser to parse the output instead of structuredoutputparser
parser = PydanticOutputParser(pydantic_object=Person)


template = PromptTemplate(
    template="""
Create an imaginary character named {name}.

{format_instructions}


Character name: {name}
""",
    input_variables=["name"],
    partial_variables={
        "format_instructions": parser.get_format_instructions()},
)
# the pydanticoutputparser will ensure that the output matches the schema defined in the Person model
# and will raise an error if it does not.

chain = template | model_google | parser

result = chain.invoke({"name": "Selena"})

print(result)
