from langchain_ollama import ChatOllama
from typing import TypedDict,Annotated,Optional,Literal
from pydantic import BaseModel, EmailStr, Field
# TypedDict is a special class in Python's typing module that allows you to specify the expected types of dictionary keys and values.

model=ChatOllama(
    model="llama3.1",
    temperature=0.5,
)


#annotated/optional typedict example, this allows you to add metadata to the fields so that ai can understand the context better
class User(BaseModel):
    name: Annotated[str, Field(default="unknown")]
    email: EmailStr
    age: Optional[int] = Field(None, ge=0, description="The age of the user, must be non-negative")

structured_output_2 = model.with_structured_output(User)

# It helps with type checking and code clarity when working with dictionaries that have a fixed structure.
user = structured_output_2.invoke(
    """
    Name: Alice Johnson
    Email: alice.johnson@example.com
    Age: 30

    Alice Johnson is a senior software engineer at TechNova Inc. She specializes in backend development, cloud architecture, and distributed systems. Alice has over 10 years of experience in the tech industry, working with Python, Go, and Kubernetes. She is passionate about mentoring junior developers and regularly contributes to open-source projects. In her free time, Alice enjoys hiking, reading science fiction novels, and experimenting with new programming languages.
    """
)
print(user)