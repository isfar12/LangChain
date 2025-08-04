# Tutorial 3: JsonOutputParser

This tutorial demonstrates using `JsonOutputParser` to get JSON-formatted output from LLMs without validation.

## Learning Objectives

- Understand how to get JSON output from LLMs
- Learn the limitations of basic JSON parsing
- See when to use JsonOutputParser vs other parsers

## Code Explanation

```python
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

chat = ChatOllama(
    model="gemma:2b",
    temperature=0.5,
)

parser = JsonOutputParser()

template = PromptTemplate(
    template="Write me about an imaginary character {name} {format_instructions}",
    input_variables=["name"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

chain = template | chat | parser

result = chain.invoke({"name": "John Doe"})
print(result)
```

## What is JsonOutputParser?

`JsonOutputParser` is designed to:
- **Extract JSON** from LLM text responses
- **Parse automatically** into Python dictionaries
- **Provide format instructions** to guide the LLM
- **Handle basic JSON** without schema validation

## Key Features

### 1. Automatic Format Instructions

```python
parser = JsonOutputParser()
instructions = parser.get_format_instructions()
print(instructions)
```

Output:
```
Return a JSON object. Do not include any explanations, only provide a RFC8259 compliant JSON response.
```

### 2. JSON Parsing

The parser automatically:
- Extracts JSON from model responses
- Converts to Python dictionaries
- Handles common JSON formatting issues

### 3. Integration with Templates

```python
template = PromptTemplate(
    template="Create a character profile for {name}. {format_instructions}",
    input_variables=["name"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)
```

## Practical Examples

### Character Generation

```python
from langchain_core.output_parsers import JsonOutputParser

parser = JsonOutputParser()

character_template = PromptTemplate(
    template="""Create a detailed character profile for {name}.
    Include: name, age, occupation, personality traits, and background.
    
    {format_instructions}""",
    input_variables=["name"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

chain = character_template | chat | parser

result = chain.invoke({"name": "Sarah Connor"})
print(result)
```

Expected output:
```python
{
    "name": "Sarah Connor",
    "age": 35,
    "occupation": "Military strategist",
    "personality": ["determined", "protective", "resourceful"],
    "background": "Former waitress turned resistance leader..."
}
```

### Product Analysis

```python
product_template = PromptTemplate(
    template="""Analyze this product: {product}
    Provide: name, category, price_range, pros, cons, rating
    
    {format_instructions}""",
    input_variables=["product"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

chain = product_template | chat | parser

result = chain.invoke({"product": "iPhone 15 Pro"})
print(result)
```

### Data Extraction

```python
extraction_template = PromptTemplate(
    template="""Extract key information from this text: {text}
    Include: main_topic, key_points, sentiment, entities
    
    {format_instructions}""",
    input_variables=["text"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)
```

## Benefits of JsonOutputParser

✅ **Structured Output** - Returns Python dictionaries
✅ **Automatic Parsing** - No manual JSON.loads() needed
✅ **Format Guidance** - Provides instructions to LLM
✅ **Flexible Schema** - No rigid structure required
✅ **Easy Integration** - Works with prompt templates

## Limitations and Challenges

❌ **No Validation** - Doesn't enforce specific fields
❌ **Inconsistent Structure** - Output format may vary
❌ **Parsing Failures** - May fail with malformed JSON
❌ **No Type Checking** - No guarantee of data types
❌ **Model Dependent** - Success depends on model capability

## Common Issues

### 1. Malformed JSON

**Problem:**
```
Model returns: "Here's the JSON: {name: 'John', age: 30}"
```

**Solution:**
- Use better models (llama3.1, mistral)
- Lower temperature for consistency
- More explicit instructions

### 2. Extra Text

**Problem:**
```
Model returns: "Sure! Here's the character: {name: 'John'} Hope this helps!"
```

**Solution:**
```python
template = PromptTemplate(
    template="""Create a character for {name}.
    
    {format_instructions}
    
    IMPORTANT: Return ONLY the JSON, no explanations.""",
    input_variables=["name"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)
```

### 3. Inconsistent Fields

**Problem:**
```
First call: {"name": "John", "age": 30}
Second call: {"character_name": "Jane", "years_old": 25}
```

**Solution:** Use `StructuredOutputParser` or `PydanticOutputParser` for consistency.

## Improved Prompting Techniques

### 1. Clear Structure Request

```python
template = PromptTemplate(
    template="""Create a character profile for {name}.

Required fields: name, age, occupation, personality, background

{format_instructions}

Character name: {name}""",
    input_variables=["name"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)
```

### 2. Example-Driven Prompts

```python
template = PromptTemplate(
    template="""Create a character like this example:
    
Example: {{"name": "Alice", "age": 28, "job": "engineer"}}

Create similar profile for: {name}

{format_instructions}""",
    input_variables=["name"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)
```

### 3. Error Handling

```python
try:
    result = chain.invoke({"name": "John Doe"})
    print(f"Success: {result}")
except Exception as e:
    print(f"Parsing failed: {e}")
    # Fallback or retry logic
```

## Best Practices

1. **Use explicit instructions** about JSON format
2. **Specify required fields** in the prompt
3. **Use consistent models** for predictable output
4. **Lower temperature** (0.1-0.3) for JSON consistency
5. **Add error handling** for parsing failures
6. **Test with various inputs** to ensure robustness

## Model Recommendations

For JSON output:
- **Best:** `llama3.1:8b`, `mistral:7b`
- **Good:** `qwen2:7b`, `codellama:7b`
- **Avoid:** Very small models that struggle with JSON format

## When to Use JsonOutputParser

✅ **Good for:**
- Quick prototyping with JSON
- Flexible data structures
- When exact schema isn't critical
- Simple JSON extraction

❌ **Consider alternatives for:**
- Production applications requiring validation
- Consistent field names needed
- Type safety requirements
- Complex nested structures

## Exercise

Create a JSON parser chain that:
1. Takes a company name as input
2. Returns company analysis with fields: name, industry, strengths, weaknesses, market_position
3. Handle parsing errors gracefully

## Next Step

Move to Tutorial 4 (`structured_output_using_parsers.py`) to learn about `StructuredOutputParser` for more reliable structured output.
