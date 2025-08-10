from pydantic import BaseModel,Field
from langchain.tools import StructuredTool


class MultiplyFormat(BaseModel):
    a: int = Field(required=True,description="The first number to multiply")
    b: int = Field(required=True,description="The second number to multiply")

def multiply(a: int, b: int) -> int:
    return a * b

multiply_tool = StructuredTool.from_function(
    func=multiply,
    name="multiply_function",
    description="Multiply two numbers",
    args_schema=MultiplyFormat
)

result=multiply_tool.invoke({"a":4,"b":7})
print(result)
print(multiply_tool.name)
print(multiply_tool.description)
print(multiply_tool.args_schema.model_json_schema())    