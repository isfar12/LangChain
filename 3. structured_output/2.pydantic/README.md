# Pydantic Structured Output

This folder demonstrates using Pydantic models for advanced structured output with LangChain. **This is the recommended approach** for most applications.

## What is Pydantic?

Pydantic is a data validation library that uses Python type annotations to validate data at runtime. It provides automatic data conversion, validation, and serialization.

## Files Overview

### `pydantic_basic.py`
Basic Pydantic model usage - shows how to define models with validation, defaults, and constraints.

### `struct_out_using_pydantic.py`
Integration with LangChain models for structured output using Pydantic models.

## Tutorial: Basic Pydantic Models

```python
from pydantic import BaseModel, EmailStr, Field
from typing import Optional, Annotated

class User(BaseModel):
    name: str = "unknown"  # Default value
    email: EmailStr        # Automatic email validation
    age: Optional[int] = Field(None, ge=0, description="Age must be non-negative")

# Create instance with validation
user = User(name="John", email="john@example.com", age=25)
print(user.dict())  # Convert to dictionary
```

## Tutorial: Advanced Pydantic Features

```python
from pydantic import BaseModel, Field
from typing import List, Literal, Annotated

class ProductReview(BaseModel):
    title: Annotated[str, Field(description="Review title")]
    rating: int = Field(ge=1, le=5, description="Rating from 1-5 stars")
    sentiment: Literal["positive", "neutral", "negative"]
    tags: List[str] = Field(default_factory=list)
    verified: bool = True

# Automatic validation
review = ProductReview(
    title="Great product!",
    rating=5,
    sentiment="positive",
    tags=["electronics", "recommended"]
)
```

## Tutorial: Pydantic with LangChain

```python
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field
from typing import List

class MovieAnalysis(BaseModel):
    title: str = Field(description="Movie title")
    genre: str = Field(description="Primary genre")
    rating: float = Field(ge=0, le=10, description="Rating out of 10")
    summary: str = Field(description="Brief summary")
    cast: List[str] = Field(description="Main cast members")

model = ChatOllama(model="llama3.1:8b", temperature=0.1)
structured_llm = model.with_structured_output(MovieAnalysis)

result = structured_llm.invoke(
    "Analyze the movie 'The Matrix' - extract title, genre, rating, summary, and main cast"
)

print(f"Title: {result.title}")
print(f"Genre: {result.genre}")
print(f"Rating: {result.rating}")
```

## Key Features

### 1. **Automatic Validation**
```python
age: int = Field(ge=0, le=120)  # Must be between 0 and 120
email: EmailStr                 # Must be valid email format
```

### 2. **Default Values**
```python
name: str = "Anonymous"
active: bool = True
tags: List[str] = Field(default_factory=list)
```

### 3. **Field Constraints**
```python
password: str = Field(min_length=8, regex=r"^(?=.*[A-Za-z])(?=.*\d)")
score: float = Field(ge=0.0, le=100.0)
```

### 4. **Rich Type System**
```python
from typing import Optional, List, Dict, Union, Literal

status: Literal["active", "inactive", "pending"]
metadata: Optional[Dict[str, Any]] = None
```

### 5. **Custom Validation**
```python
from pydantic import validator

class User(BaseModel):
    username: str
    
    @validator('username')
    def username_must_be_alphanumeric(cls, v):
        assert v.isalnum(), 'Username must be alphanumeric'
        return v
```

## Advanced Usage

### Nested Models
```python
class Address(BaseModel):
    street: str
    city: str
    zipcode: str

class User(BaseModel):
    name: str
    address: Address  # Nested model
```

### Model Configuration
```python
class User(BaseModel):
    name: str
    
    class Config:
        extra = "forbid"  # Don't allow extra fields
        validate_assignment = True  # Validate on assignment
```

## Benefits Over TypedDict

✅ **Runtime Validation**: Catches errors when data is created
✅ **Default Values**: Automatic handling of optional fields
✅ **Field Constraints**: Min/max values, string patterns, etc.
✅ **Type Conversion**: Automatic conversion between compatible types
✅ **Better Error Messages**: Clear validation error descriptions
✅ **Serialization**: Easy conversion to/from JSON, dict, etc.

## Best Practices

1. **Use descriptive field descriptions** - Helps the LLM understand what to extract
2. **Set appropriate constraints** - Use `ge`, `le`, `min_length`, etc.
3. **Provide defaults for optional fields** - Makes the model more robust
4. **Use `Literal` for enums** - Restricts values to specific options
5. **Keep models focused** - Don't make models too complex

## Common Patterns

### User Profile Extraction
```python
class UserProfile(BaseModel):
    name: str = Field(description="Full name")
    email: EmailStr = Field(description="Email address")
    age: Optional[int] = Field(None, ge=0, le=120)
    skills: List[str] = Field(default_factory=list)
```

### Sentiment Analysis
```python
class SentimentAnalysis(BaseModel):
    text: str = Field(description="Original text")
    sentiment: Literal["positive", "negative", "neutral"]
    confidence: float = Field(ge=0.0, le=1.0)
    keywords: List[str] = Field(default_factory=list)
```

### Data Extraction
```python
class Invoice(BaseModel):
    invoice_number: str
    date: str = Field(description="Invoice date in YYYY-MM-DD format")
    total: float = Field(ge=0, description="Total amount")
    items: List[str] = Field(description="List of items/services")
```

## Troubleshooting

- **Validation errors**: Check field constraints and required fields
- **Missing fields**: Add defaults or make fields optional
- **Type errors**: Ensure the LLM output matches expected types
- **Empty output**: Use more explicit prompts and lower temperature
