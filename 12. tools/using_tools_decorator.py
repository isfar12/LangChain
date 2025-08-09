from langchain_community.tools import ShellTool
from langchain_core.tools import tool
from langchain.agents import create_react_agent,AgentExecutor


# Basic Tool calling
shell=ShellTool()
result=shell.invoke("dir")
print(result)


@tool
def multiply(a: int, b: int) -> int:
    '''
    Multiplies two numbers.
    '''
    return a * b

print(multiply.invoke({"a": 2, "b": 3}))  # Should print 6
print(multiply.args_schema.model_json_schema()) # this is the format model recieves