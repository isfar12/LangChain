# JSON Schema Structured Output

This folder demonstrates using raw JSON Schema definitions for structured output with LangChain. This approach gives you maximum control over the schema specification.

## What is JSON Schema?

JSON Schema is a vocabulary that allows you to annotate and validate JSON documents. It provides a contract for JSON data and enables automatic validation, documentation, and code generation.

## Files Overview

### `basic_json.json`
A complete JSON Schema definition example showing all the key components and validation rules.

### `struct_using_json.py`
Integration with LangChain models using JSON Schema for structured output.

## Tutorial: Basic JSON Schema Structure

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "ProductReview",
  "description": "Schema for product review data",
  "type": "object",
  "properties": {
    "title": {
      "type": "string",
      "description": "Review title"
    },
    "rating": {
      "type": "number",
      "minimum": 1,
      "maximum": 5
    }
  },
  "required": ["title", "rating"]
}
```

## Tutorial: JSON Schema with LangChain

```python
from langchain_ollama import ChatOllama
import json

# Define schema as Python dictionary (not JSON string!)
schema = {
    "title": "PersonInfo",
    "type": "object",
    "properties": {
        "name": {
            "type": "string",
            "description": "Full name"
        },
        "age": {
            "type": "integer",
            "minimum": 0,
            "maximum": 120
        },
        "email": {
            "type": "string",
            "format": "email"
        },
        "skills": {
            "type": "array",
            "items": {"type": "string"}
        }
    },
    "required": ["name", "email"],
    "additionalProperties": False
}

model = ChatOllama(model="llama3.1:8b", temperature=0.1)
structured_llm = model.with_structured_output(schema)

result = structured_llm.invoke(
    "Extract info: John Doe, 30 years old, john@email.com, knows Python and JavaScript"
)

print(json.dumps(result, indent=2))
```

## JSON Schema Components

### 1. **Basic Types**
```json
{
  "name": {"type": "string"},
  "age": {"type": "integer"},
  "price": {"type": "number"},
  "active": {"type": "boolean"},
  "tags": {"type": "array"},
  "metadata": {"type": "object"}
}
```

### 2. **String Validation**
```json
{
  "email": {
    "type": "string",
    "format": "email"
  },
  "username": {
    "type": "string",
    "minLength": 3,
    "maxLength": 20,
    "pattern": "^[a-zA-Z0-9_]+$"
  }
}
```

### 3. **Number Constraints**
```json
{
  "age": {
    "type": "integer",
    "minimum": 0,
    "maximum": 120
  },
  "price": {
    "type": "number",
    "minimum": 0,
    "exclusiveMaximum": 1000.00
  }
}
```

### 4. **Array Definitions**
```json
{
  "tags": {
    "type": "array",
    "items": {"type": "string"},
    "minItems": 1,
    "maxItems": 10,
    "uniqueItems": true
  }
}
```

### 5. **Enum Values**
```json
{
  "status": {
    "type": "string",
    "enum": ["active", "inactive", "pending"]
  }
}
```

### 6. **Object Properties**
```json
{
  "address": {
    "type": "object",
    "properties": {
      "street": {"type": "string"},
      "city": {"type": "string"},
      "zipcode": {"type": "string"}
    },
    "required": ["street", "city"]
  }
}
```

## Advanced JSON Schema Features

### Conditional Logic
```json
{
  "if": {
    "properties": {"type": {"const": "premium"}}
  },
  "then": {
    "required": ["premium_features"]
  }
}
```

### Multiple Types
```json
{
  "value": {
    "anyOf": [
      {"type": "string"},
      {"type": "number"}
    ]
  }
}
```

### Complex Validation
```json
{
  "password": {
    "type": "string",
    "minLength": 8,
    "pattern": "^(?=.*[a-z])(?=.*[A-Z])(?=.*\\d)(?=.*[@$!%*?&])[A-Za-z\\d@$!%*?&]"
  }
}
```

## Complete Example: E-commerce Product

```python
ecommerce_schema = {
    "title": "Product",
    "type": "object",
    "properties": {
        "name": {
            "type": "string",
            "minLength": 1,
            "description": "Product name"
        },
        "category": {
            "type": "string",
            "enum": ["electronics", "clothing", "books", "home", "sports"]
        },
        "price": {
            "type": "number",
            "minimum": 0,
            "description": "Price in USD"
        },
        "rating": {
            "type": "number",
            "minimum": 0,
            "maximum": 5
        },
        "features": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Key product features"
        },
        "in_stock": {
            "type": "boolean",
            "default": True
        },
        "specifications": {
            "type": "object",
            "properties": {
                "weight": {"type": "string"},
                "dimensions": {"type": "string"},
                "color": {"type": "string"}
            }
        }
    },
    "required": ["name", "category", "price"],
    "additionalProperties": False
}
```

## Key Advantages

✅ **Maximum Flexibility**: Define any validation rule possible
✅ **Standard Format**: JSON Schema is an industry standard
✅ **Rich Validation**: Complex constraints and conditional logic
✅ **Documentation**: Self-documenting schemas
✅ **Tool Support**: Many tools support JSON Schema

## Limitations

❌ **Verbose**: More code than Pydantic for simple cases
❌ **Python Dictionary**: Must convert JSON string to dict for LangChain
❌ **No IDE Support**: Less autocomplete compared to Pydantic
❌ **Complex Syntax**: Learning curve for advanced features

## Best Practices

1. **Use Python dictionaries** - Don't pass JSON strings to `with_structured_output()`
2. **Add descriptions** - Help the LLM understand each field
3. **Set constraints** - Use min/max, patterns, enums to limit output
4. **Make fields optional** - Use defaults or don't include in `required`
5. **Test thoroughly** - Validate with different inputs

## Common Patterns

### Data Extraction
```python
extraction_schema = {
    "type": "object",
    "properties": {
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "type": {"enum": ["person", "organization", "location"]},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1}
                }
            }
        }
    }
}
```

### Classification
```python
classification_schema = {
    "type": "object",
    "properties": {
        "category": {"enum": ["spam", "ham", "promotional"]},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "reasoning": {"type": "string"}
    },
    "required": ["category", "confidence"]
}
```

## When to Use JSON Schema

✅ **Good for:**
- Complex validation requirements
- When you need maximum control
- Integration with existing JSON Schema tools
- API documentation generation

❌ **Consider Pydantic instead for:**
- Python-centric applications
- Simpler validation needs
- Better IDE support requirements
- Faster development
