# Tutorial 4: StructuredOutputParser

This tutorial demonstrates using `StructuredOutputParser` with `ResponseSchema` for more reliable structured output with defined schemas.

## Learning Objectives

- Understand how to define output schemas with ResponseSchema
- Learn the benefits of structured parsing over basic JSON
- See how to create consistent, predictable outputs

## Code Explanation

```python
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

chat = ChatOllama(
    model="gemma:2b",
    temperature=0.5,
)

# Define schema with ResponseSchema
schema = [
    ResponseSchema(name="name", description="The name of the character"),
    ResponseSchema(name="age", description="The age of the character"),
    ResponseSchema(name="description", description="A brief description of the character"),
]

parser = StructuredOutputParser.from_response_schemas(schema)

template = PromptTemplate(
    template="Write me about an imaginary character named {name} {format_instructions}",
    input_variables=["name"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

chain = template | chat | parser

result = chain.invoke({"name": "John Doe"})
print(result)
```

## What is StructuredOutputParser?

`StructuredOutputParser` provides:
- **Schema Definition** - Explicit field specifications
- **Consistent Structure** - Same fields every time
- **Better Instructions** - Detailed format guidance
- **Type Hints** - Field descriptions for the LLM

## Key Components

### 1. ResponseSchema

`ResponseSchema` defines individual fields:

```python
ResponseSchema(
    name="field_name",           # Required: field identifier
    description="Field purpose"  # Required: helps LLM understand
)
```

### 2. Schema Creation

```python
schema = [
    ResponseSchema(name="title", description="Article title"),
    ResponseSchema(name="summary", description="Brief summary"),
    ResponseSchema(name="category", description="Content category"),
]
```

### 3. Parser Creation

```python
parser = StructuredOutputParser.from_response_schemas(schema)
```

## Advanced Schema Examples

### Product Review Schema

```python
review_schema = [
    ResponseSchema(name="product_name", description="Name of the product"),
    ResponseSchema(name="rating", description="Rating from 1-5 stars"),
    ResponseSchema(name="pros", description="List of positive aspects"),
    ResponseSchema(name="cons", description="List of negative aspects"),
    ResponseSchema(name="recommendation", description="Would you recommend this product"),
    ResponseSchema(name="price_value", description="Is the price justified"),
]

parser = StructuredOutputParser.from_response_schemas(review_schema)
```

### News Article Analysis

```python
news_schema = [
    ResponseSchema(name="headline", description="Main headline"),
    ResponseSchema(name="summary", description="Article summary in 2-3 sentences"),
    ResponseSchema(name="sentiment", description="Overall sentiment: positive, negative, or neutral"),
    ResponseSchema(name="key_entities", description="Important people, places, organizations mentioned"),
    ResponseSchema(name="category", description="News category: politics, technology, sports, etc."),
    ResponseSchema(name="credibility", description="Credibility assessment: high, medium, low"),
]
```

### Company Analysis

```python
company_schema = [
    ResponseSchema(name="company_name", description="Full company name"),
    ResponseSchema(name="industry", description="Primary industry sector"),
    ResponseSchema(name="founded", description="Year founded"),
    ResponseSchema(name="headquarters", description="Location of headquarters"),
    ResponseSchema(name="market_cap", description="Approximate market capitalization"),
    ResponseSchema(name="key_products", description="Main products or services"),
    ResponseSchema(name="competitors", description="Major competitors"),
    ResponseSchema(name="recent_news", description="Recent significant developments"),
]
```

## Format Instructions

The parser generates detailed instructions:

```python
parser = StructuredOutputParser.from_response_schemas(schema)
instructions = parser.get_format_instructions()
print(instructions)
```

Example output:
```
The output should be a markdown code snippet formatted in the following schema, including the leading and trailing "```json" and "```":

```json
{
    "name": string  // The name of the character
    "age": string  // The age of the character  
    "description": string  // A brief description of the character
}
```
```

## Benefits Over JsonOutputParser

✅ **Consistent Fields** - Same structure every time
✅ **Field Documentation** - Descriptions guide the LLM
✅ **Better Instructions** - More detailed format guidance
✅ **Predictable Output** - Reduces parsing failures
✅ **Schema Validation** - Basic structure checking

## Practical Implementation

### Complete Character Generator

```python
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

# Define comprehensive character schema
character_schema = [
    ResponseSchema(name="name", description="Character's full name"),
    ResponseSchema(name="age", description="Character's age in years"),
    ResponseSchema(name="occupation", description="Character's job or profession"),
    ResponseSchema(name="personality", description="Key personality traits"),
    ResponseSchema(name="background", description="Character's backstory"),
    ResponseSchema(name="goals", description="Character's main objectives"),
    ResponseSchema(name="relationships", description="Important relationships"),
]

parser = StructuredOutputParser.from_response_schemas(character_schema)

template = PromptTemplate(
    template="""Create a detailed character profile for {character_type} named {name}.

Make the character interesting and well-developed.

{format_instructions}

Character Type: {character_type}
Character Name: {name}""",
    input_variables=["character_type", "name"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

chain = template | chat | parser

# Generate character
result = chain.invoke({
    "character_type": "space explorer", 
    "name": "Captain Nova"
})

print(f"Name: {result['name']}")
print(f"Age: {result['age']}")
print(f"Occupation: {result['occupation']}")
print(f"Background: {result['background']}")
```

### Error Handling

```python
def safe_character_generation(character_type, name):
    try:
        result = chain.invoke({
            "character_type": character_type,
            "name": name
        })
        return result
    except Exception as e:
        print(f"Parsing failed: {e}")
        return {
            "name": name,
            "age": "Unknown",
            "occupation": "Unknown",
            "personality": "To be determined",
            "background": "Mysterious",
            "goals": "Unknown",
            "relationships": "None specified"
        }
```

## Best Practices

### 1. Clear Field Descriptions

**Good:**
```python
ResponseSchema(name="sentiment", description="Overall sentiment: positive, negative, or neutral")
```

**Poor:**
```python
ResponseSchema(name="sentiment", description="Sentiment")
```

### 2. Logical Field Names

**Good:**
```python
ResponseSchema(name="publication_date", description="When the article was published")
```

**Poor:**
```python
ResponseSchema(name="date", description="Some date")
```

### 3. Comprehensive Schemas

Include all fields you need upfront:

```python
complete_schema = [
    ResponseSchema(name="title", description="Article title"),
    ResponseSchema(name="author", description="Article author"),
    ResponseSchema(name="date", description="Publication date"),
    ResponseSchema(name="summary", description="Brief summary"),
    ResponseSchema(name="category", description="Content category"),
    ResponseSchema(name="word_count", description="Approximate word count"),
]
```

## Limitations

❌ **No Type Validation** - Still doesn't enforce data types
❌ **Basic Validation** - Only checks field presence
❌ **String Outputs** - All fields returned as strings
❌ **No Constraints** - Can't enforce value ranges or formats

## Common Issues

### 1. Missing Fields

**Problem:** Parser expects all fields but LLM skips some

**Solution:**
```python
# Make fields optional in prompt
template = PromptTemplate(
    template="""Create character for {name}.
    
If information is unknown, use "Unknown" or "Not specified".
    
{format_instructions}""",
    input_variables=["name"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)
```

### 2. Inconsistent Field Values

**Problem:** LLM returns different formats for same field

**Solution:**
```python
ResponseSchema(
    name="rating", 
    description="Rating as a number from 1 to 5 (just the number, no text)"
)
```

### 3. Parsing Failures

**Problem:** LLM doesn't follow JSON format exactly

**Solution:**
- Use more reliable models (llama3.1:8b, mistral:7b)
- Lower temperature for consistency
- Add explicit format reminders in prompt

## Model Recommendations

For structured output:
- **Best:** `llama3.1:8b`, `mistral:7b`
- **Good:** `qwen2:7b`, `codellama:7b`
- **Acceptable:** `phi3:medium`
- **Avoid:** Models smaller than 7B parameters

## When to Use StructuredOutputParser

✅ **Good for:**
- Consistent field requirements
- When you need the same structure every time
- Moderate complexity schemas
- Better than basic JSON parsing

❌ **Consider PydanticOutputParser for:**
- Type validation requirements
- Complex data constraints
- Production applications
- Advanced validation needs

## Exercise

Create a structured parser for movie analysis with fields:
- movie_title
- director
- genre
- release_year
- plot_summary
- main_characters
- critical_rating
- audience_appeal

Test it with different movie titles and ensure consistent output.

## Next Step

Move to Tutorial 5 (`structured_output_using_pydantic.py`) to learn about `PydanticOutputParser` for advanced validation and type safety.
