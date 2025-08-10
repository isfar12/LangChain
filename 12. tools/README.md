# LangChain Tools: Complete Tutorial Guide

## Overview
This section covers LangChain Tools - components that allow AI models to interact with external systems, APIs, and perform specific functions. Tools extend the capabilities of LLMs beyond text generation to actionable operations.

## Table of Contents
1. [Basic Tool Creation with Decorators](#1-basic-tool-creation-with-decorators)
2. [Structured Tools with Pydantic](#2-structured-tools-with-pydantic)
3. [Interactive Tool Calling](#3-interactive-tool-calling)
4. [Project: AI Currency Converter Agent](#4-project-ai-currency-converter-agent)

---

## 1. Basic Tool Creation with Decorators

**File: `1. using_tools_decorator.py`**

### What You'll Learn:
- Create tools using the `@tool` decorator
- Use built-in tools like ShellTool
- Understand tool schemas and arguments

### Step 1: Using Built-in Tools
```python
from langchain_community.tools import ShellTool

shell = ShellTool()
result = shell.invoke("dir")
print(result)
```
- **ShellTool**: Executes system commands
- **invoke()**: Runs the tool with given input

### Step 2: Creating Custom Tools
```python
from langchain_core.tools import tool

@tool
def multiply(a: int, b: int) -> int:
    '''Multiplies two numbers.'''
    return a * b

print(multiply.invoke({"a": 2, "b": 3}))  # Output: 6
print(multiply.args_schema.model_json_schema())  # Shows tool format
```
- **@tool decorator**: Converts function to LangChain tool
- **Docstring**: Becomes tool description
- **args_schema**: Defines expected input format

---

## 2. Structured Tools with Pydantic

**File: `2. using_pydantic.py`**

### What You'll Learn:
- Define tool schemas using Pydantic models
- Add detailed parameter descriptions
- Create more robust tool validation

### Step 1: Define Pydantic Schema
```python
from pydantic import BaseModel, Field

class MultiplyFormat(BaseModel):
    a: int = Field(required=True, description="The first number to multiply")
    b: int = Field(required=True, description="The second number to multiply")
```
- **BaseModel**: Pydantic base class for data validation
- **Field**: Adds metadata and descriptions to parameters

### Step 2: Create Structured Tool
```python
from langchain.tools import StructuredTool

def multiply(a: int, b: int) -> int:
    return a * b

multiply_tool = StructuredTool.from_function(
    func=multiply,
    name="multiply_function",
    description="Multiply two numbers",
    args_schema=MultiplyFormat
)

result = multiply_tool.invoke({"a": 4, "b": 7})
print(result)  # Output: 28
```
- **StructuredTool**: More controlled tool creation
- **from_function()**: Converts regular function to tool
- **args_schema**: Links Pydantic model for validation

---

## 3. Interactive Tool Calling

**File: `3. basic_tool_calling.ipynb`**

### What You'll Learn:
- Bind tools to AI models
- Handle tool calls and responses
- Create complete conversational flows
- Manage message chains for tool interactions

### Step 1: Initialize Model and Bind Tools
```python
from langchain_ollama import ChatOllama

chat = ChatOllama(model="llama3.2:latest", temperature=.5)
llm_with_tools = chat.bind_tools([multiply])
```
- **bind_tools()**: Connects tools to the LLM
- **Temperature**: Controls randomness in responses

### Step 2: Test Tool Detection
```python
response = llm_with_tools.invoke("Can you multiply 2 and 3?")
tool_calls = response.tool_calls
print(tool_calls)  # Shows which tools AI wants to use
```
- **tool_calls**: List of tools AI decided to execute
- AI automatically detects when tools are needed

### Step 3: Execute Tools Manually
```python
result = multiply.invoke(tool_calls[0])
print(result)  # Actual tool execution result
```

### Step 4: Complete Conversational Flow
The notebook demonstrates the complete 8-step process:

1. **Create Human Message**: Initial user request
2. **AI Response**: LLM processes and identifies needed tools
3. **Message Chain**: Add both messages to conversation
4. **Extract Tool Calls**: Get tool parameters from AI response
5. **Execute Tools**: Run tools with extracted parameters
6. **Add Tool Results**: Include results in message chain
7. **Final AI Response**: AI synthesizes complete answer
8. **Display Result**: Show final response to user

#### Complete Implementation Code:
```python
from langchain_core.messages import HumanMessage

# Step 1: Create initial human message
message = HumanMessage("What is the multiplication of 2 and 10?")
messages = [message]

# Step 2: Get AI response with tool calls
llm_message = llm_with_tools.invoke(messages)
print("AI Response:", llm_message)
print("Tool Calls:", llm_message.tool_calls)

# Step 3: Add AI message to conversation
messages.append(llm_message)

# Step 4: Extract and execute tool calls
if llm_message.tool_calls:
    for tool_call in llm_message.tool_calls:
        # Step 5: Execute the tool with extracted parameters
        tool_result = multiply.invoke(tool_call)
        print("Tool Result:", tool_result)
        
        # Step 6: Add tool result to message chain
        messages.append(tool_result)

# Step 7: Get final AI response with tool results
final_response = llm_with_tools.invoke(messages)

# Step 8: Display final result
print("Final Answer:", final_response.content)
```

#### Alternative Single-Flow Implementation:
```python
# Simplified workflow for single tool usage
message = [HumanMessage("What is the multiplication of 2 and 10?")]
llm_message = llm_with_tools.invoke(message)
message.append(llm_message)

# Execute tool
tool_calls = multiply.invoke(llm_message.tool_calls[0])
message.append(tool_calls)

# Get final response
final_response = llm_with_tools.invoke(message)
print(final_response.content)
```

#### Understanding the Message Flow:
```python
# Examine message structure at each step
print("1. Initial Messages:", messages)
print("2. After AI Response:", len(messages), "messages")
print("3. After Tool Execution:", len(messages), "messages")
print("4. Final Response Content:", final_response.content)
```

#### Exploring Tool Integration Features:
```python
# Check tool schemas and metadata
print("Tool Name:", multiply.name)
print("Tool Description:", multiply.description)
print("Tool Arguments Schema:", multiply.args_schema.model_json_schema())

# Examine AI's tool selection process
response = llm_with_tools.invoke("Can you multiply 2 and 3?")
print("AI chose tools:", [call['name'] for call in response.tool_calls])
print("With arguments:", [call['args'] for call in response.tool_calls])

# Test different queries to see tool selection
queries = [
    "What is 5 times 7?",
    "Can you help me multiply two numbers?", 
    "Calculate the product of 12 and 8",
    "Hello, how are you?"  # No tools needed
]

for query in queries:
    response = llm_with_tools.invoke(query)
    tools_used = len(response.tool_calls) if response.tool_calls else 0
    print(f"Query: '{query}' → Tools used: {tools_used}")
```

#### Working with Multiple Tools:
```python
# Add more tools to demonstrate coordination
from langchain_community.tools import ShellTool
import os

shell = ShellTool()

# Create a multi-tool environment
@tool
def add_numbers(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b

@tool
def divide_numbers(a: float, b: float) -> float:
    """Divide first number by second number."""
    if b == 0:
        return "Error: Division by zero"
    return a / b

# Bind multiple tools
multi_tool_llm = chat.bind_tools([multiply, add_numbers, divide_numbers, shell])

# Test complex query requiring multiple tools
complex_query = "First multiply 6 by 4, then add 5 to the result, then divide by 3"
response = multi_tool_llm.invoke(complex_query)

print("AI Response:", response.content)
print("Tools to use:", [call['name'] for call in response.tool_calls])

# Execute tools in sequence
messages = [HumanMessage(complex_query)]
ai_msg = multi_tool_llm.invoke(messages)
messages.append(ai_msg)

# Process each tool call
results = {}
for tool_call in ai_msg.tool_calls:
    if tool_call['name'] == 'multiply':
        result = multiply.invoke(tool_call['args'])
        results['multiply'] = result
    elif tool_call['name'] == 'add_numbers':
        result = add_numbers.invoke(tool_call['args'])  
        results['add'] = result
    elif tool_call['name'] == 'divide_numbers':
        result = divide_numbers.invoke(tool_call['args'])
        results['divide'] = result
    
    messages.append(result)

# Final response with all tool results
final = multi_tool_llm.invoke(messages)
print("Final coordinated response:", final.content)
```

### Key Concepts:
- **Message Chaining**: Maintaining conversation context
- **Tool Call Extraction**: Getting parameters from AI responses
- **Result Integration**: Combining tool outputs with AI reasoning
- **Conversational Flow**: Creating natural interactions

---

## 4. Project: AI Currency Converter Agent

**File: `4. Project_Currency_convert.ipynb`**

### Project Overview
Build a complete AI agent that can:
- Fetch real-time exchange rates from external APIs
- Convert currencies using live data
- Handle complex multi-step requests
- Coordinate multiple tools intelligently
- Prevent AI hallucination of critical data values

### Step 1: API Testing and Understanding
```python
import requests
url = "https://v6.exchangerate-api.com/v6/API_KEY/pair/USD/BDT"
response = requests.get(url)
data = response.json()
```
First, we test the external API to understand:
- Data structure returned by the API
- How to construct proper API calls
- What fields contain the conversion rates

### Step 2: Initial Tool Creation
```python
@tool
def get_exchange_rate(from_currency: str, to_currency: str) -> float:
    """Get the exchange rate from one currency to another."""
    url = f"https://v6.exchangerate-api.com/v6/API_KEY/pair/{from_currency}/{to_currency}"
    response = requests.get(url)
    data = response.json()
    return data["conversion_rate"]

@tool
def convert_currency(amount: float, conversion_rate: float) -> float:
    """Convert currency using conversion rate."""
    return amount * conversion_rate
```

### Step 3: Model Setup and Tool Binding
```python
from langchain_ollama import ChatOllama
llm = ChatOllama(model="llama3.2:latest", temperature=0.7)
llm_with_tools = llm.bind_tools([get_exchange_rate, convert_currency])
```

### Step 4: Problem Discovery - AI Hallucination
When testing with: *"What is the exchange rate from USD to BDT and convert 10 USD to BDT"*

**Problem Found**: The AI auto-generates conversion_rate values instead of waiting for real API data!

```python
# AI incorrectly calls tools like this:
tool_calls = [
    {"name": "get_exchange_rate", "args": {"from_currency": "USD", "to_currency": "BDT"}},
    {"name": "convert_currency", "args": {"amount": 10, "conversion_rate": 1.0}}  # WRONG!
]
```

### Step 5: The Solution - InjectedToolArg
```python
from langchain_core.tools import InjectedToolArg
from typing import Annotated

@tool
def convert_currency(amount: float, conversion_rate: Annotated[float, InjectedToolArg]) -> float:
    """Convert currency using conversion rate."""
    return amount * conversion_rate
```

**Key Innovation**: `InjectedToolArg` tells the AI:
- Don't auto-generate this parameter
- This value will be provided during execution
- Wait for external data before proceeding

### Step 6: Coordinated Tool Execution
The agent now works in proper sequence:
1. **Fetch Rate**: Get real exchange rate from API
2. **Extract Value**: Pull conversion rate from API response
3. **Convert Currency**: Use real rate for accurate conversion
4. **Generate Response**: Provide complete answer with real data

### Step 7: Complete Workflow Implementation
Here's the complete currency conversion workflow with actual code from the notebook:

#### Initial Test (Problem Demonstration):
```python
from langchain_core.messages import HumanMessage

# Step 1: Create user request
messages = [HumanMessage("What is the exchange rate from USD to BDT and using that rate, convert 10 USD to BDT")]

# Step 2: AI response (shows the problem)
ai_response = llm_with_tools.invoke(messages)
print("AI Response:", ai_response)
print("Tool Calls:", ai_response.tool_calls)

# Output shows AI auto-generating conversion_rate instead of waiting for API data
```

#### Fixed Implementation with InjectedToolArg:
```python
from langchain_core.tools import tool, InjectedToolArg
from typing import Annotated
import requests

@tool
def get_exchange_rate(from_currency: str, to_currency: str) -> dict:
    """Get the exchange rate from one currency to another."""
    url = f"https://v6.exchangerate-api.com/v6/22dd33f5efd4cebd27af814d/pair/{from_currency}/{to_currency}"
    response = requests.get(url)
    data = response.json()
    return data  # Returns full response including conversion_rate

@tool
def convert_currency(amount: float, conversion_rate: Annotated[float, InjectedToolArg]) -> float:
    """Convert currency using conversion rate."""
    return amount * conversion_rate

# Rebind tools with fixed version
llm_with_tools = llm.bind_tools([get_exchange_rate, convert_currency])
```

#### Complete Orchestration Logic:
```python
# Step 1: User makes request
messages = [HumanMessage("What is the exchange rate from USD to BDT and convert 10 USD to BDT")]

# Step 2: AI identifies tools needed
ai_response = llm_with_tools.invoke(messages)
messages.append(ai_response)

# Step 3: Process each tool call
for tool_call in ai_response.tool_calls:
    if tool_call['name'] == 'get_exchange_rate':
        # Execute exchange rate tool
        rate_data = get_exchange_rate.invoke(tool_call['args'])
        print(f"Exchange Rate Data: {rate_data}")
        
        # Store the conversion rate for next tool
        conversion_rate = rate_data['conversion_rate']
        
    elif tool_call['name'] == 'convert_currency':
        # Execute conversion with injected rate
        converted_amount = convert_currency.invoke({
            'amount': tool_call['args']['amount'],
            'conversion_rate': conversion_rate  # Injected from previous tool
        })
        print(f"Converted Amount: {converted_amount}")

# Step 4: Create tool results and add to messages
# (Tool result creation code from notebook)

# Step 5: Final AI response
final_answer = llm_with_tools.invoke(messages)
print("Final Answer:", final_answer.content)
```

#### Testing Individual Components:
```python
# Test exchange rate tool independently
rate_result = get_exchange_rate.invoke({"from_currency": "USD", "to_currency": "BDT"})
print("Rate Result:", rate_result)
print("Conversion Rate:", rate_result["conversion_rate"])

# Test conversion tool with real rate
conversion_result = convert_currency.invoke({
    "amount": 10, 
    "conversion_rate": rate_result["conversion_rate"]
})
print("Conversion Result:", conversion_result)
```

#### Error Handling and Validation:
```python
# Add error handling for API calls
@tool
def get_exchange_rate_safe(from_currency: str, to_currency: str) -> dict:
    """Get exchange rate with error handling."""
    try:
        url = f"https://v6.exchangerate-api.com/v6/API_KEY/pair/{from_currency}/{to_currency}"
        response = requests.get(url)
        response.raise_for_status()  # Raises exception for bad status codes
        data = response.json()
        
        if data.get("result") == "success":
            return data
        else:
            return {"error": "API returned error", "details": data}
            
    except requests.RequestException as e:
        return {"error": "Network error", "details": str(e)}
    except Exception as e:
        return {"error": "Unexpected error", "details": str(e)}
```

#### Advanced: Multi-Currency Conversion Chain:
```python
# Example of chaining multiple conversions
def multi_currency_conversion(base_amount, from_curr, to_currencies):
    results = {}
    
    for to_curr in to_currencies:
        # Get rate
        rate_data = get_exchange_rate.invoke({
            "from_currency": from_curr, 
            "to_currency": to_curr
        })
        
        # Convert currency
        converted = convert_currency.invoke({
            "amount": base_amount,
            "conversion_rate": rate_data["conversion_rate"]
        })
        
        results[to_curr] = {
            "rate": rate_data["conversion_rate"],
            "converted_amount": converted
        }
    
    return results

# Usage example
results = multi_currency_conversion(100, "USD", ["BDT", "EUR", "GBP"])
print("Multi-currency results:", results)
```

#### Final Agent Orchestration (From Notebook):
```python
import json

# Complete agent workflow from the actual notebook
messages = [HumanMessage("What is the exchange rate from USD to BDT and convert 10 USD to BDT")]
ai_response = llm_with_tools.invoke(messages)
messages.append(ai_response)

# Smart orchestration logic
for tool_call in ai_response.tool_calls:
    if tool_call["name"] == "get_exchange_rate":
        # Execute exchange rate tool first
        tool_message1 = get_exchange_rate.invoke(tool_call)
        exchange_rate = json.loads(tool_message1.content)["conversion_rate"]
        messages.append(tool_message1)
        
    if tool_call["name"] == "convert_currency":
        # Inject real exchange rate into conversion tool
        tool_call["args"]["conversion_rate"] = exchange_rate
        tool_message2 = convert_currency.invoke(tool_call)
        messages.append(tool_message2)

# Generate final natural language response
answer = llm_with_tools.invoke(messages)
print("Final Answer:", answer.content)

# View complete message chain
print("Complete Message Chain:")
for i, msg in enumerate(messages):
    print(f"{i+1}. {type(msg).__name__}: {msg}")
```

#### Key Orchestration Features:
1. **Sequential Execution**: Exchange rate → Currency conversion
2. **Data Injection**: Real API data replaces AI-generated values  
3. **Message Management**: Complete conversation context maintained
4. **Natural Response**: AI synthesizes tool results into user-friendly answer
5. **Error Recovery**: Handles API failures and network issues

#### Production Deployment Considerations:
```python
# Add rate limiting and caching
from functools import lru_cache
import time

@lru_cache(maxsize=100)
@tool
def get_exchange_rate_cached(from_currency: str, to_currency: str, cache_time: int = None) -> dict:
    """Cached exchange rate with timestamp for rate limiting."""
    # Add timestamp to cache key
    current_hour = int(time.time() / 3600)  # Cache for 1 hour
    return get_exchange_rate(from_currency, to_currency)

# Add authentication and API key management
import os
API_KEY = os.getenv('EXCHANGE_RATE_API_KEY', 'your-default-key')

@tool  
def get_exchange_rate_secure(from_currency: str, to_currency: str) -> dict:
    """Secure exchange rate fetching with proper API key management."""
    url = f"https://v6.exchangerate-api.com/v6/{API_KEY}/pair/{from_currency}/{to_currency}"
    # Rest of implementation...
```

### Architecture Components:
- **External API Integration**: Real-time exchange rate data
- **Tool Coordination**: Sequential execution with data passing
- **Parameter Injection**: Preventing AI hallucination
- **Error Handling**: Robust API interaction
- **Natural Language Interface**: User-friendly conversations

### Key Technical Achievements:
1. **Data Integrity**: Ensures real API data is used, not AI-generated values
2. **Tool Orchestration**: Coordinates multiple tools in proper sequence
3. **Parameter Flow**: Manages data passing between tools
4. **Conversational AI**: Maintains natural language interaction
5. **Production Ready**: Handles real-world API integration

### Real-World Applications:
- Financial trading platforms with live data
- E-commerce sites with dynamic pricing
- Travel apps with current exchange rates
- International business tools
- Any system requiring external data coordination

This project demonstrates the complete process of building sophisticated AI agents that can:
- Interact with external services reliably
- Coordinate multiple operations
- Prevent common AI pitfalls like data hallucination
- Provide accurate, real-time responses

---

## Key Concepts Explained

### Tool Components:
- **Function**: The actual operation to perform
- **Schema**: Defines input parameters and types
- **Description**: Helps AI understand when to use the tool
- **Name**: Unique identifier for the tool

### Integration Flow:
1. **Tool Creation**: Define what the tool does
2. **Tool Binding**: Connect to AI model
3. **Query Processing**: AI decides which tools to use
4. **Tool Execution**: Run selected tools with parameters
5. **Response Generation**: AI synthesizes final answer

### Best Practices:
- Use clear, descriptive function names
- Provide detailed docstrings
- Add proper type hints
- Handle errors gracefully
- Validate input parameters

---

## Prerequisites
- Basic Python knowledge
- Understanding of APIs and HTTP requests
- Familiarity with LangChain basics

## Next Steps
After completing this tutorial, you'll be able to:
- Create custom tools for any functionality
- Integrate tools with AI models
- Build complete AI agents
- Handle real-world API interactions