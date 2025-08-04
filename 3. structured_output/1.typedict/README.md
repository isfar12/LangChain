# TypedDict Structured Output

This folder demonstrates using Python's `TypedDict` for basic structured output with LangChain.

## What is TypedDict?

`TypedDict` is a Python typing construct that allows you to specify the expected types for dictionary keys. It provides type hints for dictionaries with a fixed set of keys.

## Files Overview

### `typedict_basic.py`

Basic TypedDict usage without LangChain - shows how to define and use TypedDict classes.

### `struct_out_using_typedict.py`

Integration with LangChain models for structured output using TypedDict.

## Tutorial: Basic TypedDict

```python
from typing import TypedDict

# Define a TypedDict class
class Person(TypedDict):
    name: str
    age: int
    is_student: bool

# Create an instance
person: Person = {
    "name": "Alice",
    "age": 30,
    "is_student": False
}
```

## Tutorial: TypedDict with LangChain

```python
from langchain_ollama import ChatOllama
from typing import TypedDict

class UserProfile(TypedDict):
    name: str
    email: str
    age: int
    interests: list[str]

model = ChatOllama(model="llama3.1:8b", temperature=0.1)

# Use with_structured_output()
structured_llm = model.with_structured_output(UserProfile)

result = structured_llm.invoke(
    "Extract user info: John Doe, john@email.com, 25 years old, likes programming and hiking"
)

print(result)  # Returns a dictionary matching UserProfile structure
```

## Key Features

- **Simple Type Hints**: Basic type specification for dictionary keys
- **No Validation**: TypedDict doesn't validate data at runtime
- **IDE Support**: Good autocomplete and type checking in IDEs
- **Lightweight**: Minimal overhead, just type hints

## Limitations

- **No Runtime Validation**: Won't catch type errors at runtime
- **No Default Values**: Can't specify default values for fields
- **No Constraints**: Can't add validation rules (min/max values, etc.)
- **Basic Types Only**: Limited to simple Python types

## When to Use TypedDict

✅ **Good for:**

- Simple dictionary structures
- Basic type hints
- When you don't need validation
- Prototyping and quick implementations

❌ **Not ideal for:**

- Complex validation requirements
- Default values needed
- Field constraints (min/max, regex, etc.)
- Production applications requiring robust validation

## Next Steps

For more advanced structured output with validation, check out the `2.pydantic/` folder which provides:

- Runtime validation
- Default values
- Field constraints
- Better error handling
