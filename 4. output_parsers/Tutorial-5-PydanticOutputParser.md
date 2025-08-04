# Tutorial 5: PydanticOutputParser

This tutorial demonstrates using `PydanticOutputParser` for advanced output parsing with full validation, type checking, and error handling.

## Learning Objectives

- Understand Pydantic models for output validation
- Learn advanced field constraints and validation
- Master error handling for production applications
- Compare all parsing approaches

## Code Explanation

```python
from langchain_core.output_parsers import PydanticOutputParser
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field

chat = ChatOllama(
    model="phi3:latest",
    temperature=0.1,
)

# Define Pydantic model with validation
class Person(BaseModel):
    name: str = Field(description="The name of the character")
    age: int = Field(gt=20, description="The age of the character")
    description: str = Field(description="A brief description of the character")

parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(
    template="""Create an imaginary character named {name}.

{format_instructions}

Character name: {name}""",
    input_variables=["name"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

chain = template | chat | parser

result = chain.invoke({"name": "Selena"})
print(result)
```

## Why PydanticOutputParser?

`PydanticOutputParser` is the **most advanced** parser offering:
- **Type Validation** - Ensures correct data types
- **Field Constraints** - Validates value ranges and formats
- **Automatic Conversion** - Converts compatible types
- **Rich Error Messages** - Detailed validation feedback
- **Production Ready** - Robust error handling

## Pydantic Model Features

### 1. Basic Field Types

```python
from pydantic import BaseModel, Field
from typing import List, Optional

class Product(BaseModel):
    name: str                           # Required string
    price: float                        # Required float
    in_stock: bool                      # Required boolean
    tags: List[str]                     # Required list of strings
    description: Optional[str] = None   # Optional string
```

### 2. Field Constraints

```python
class User(BaseModel):
    name: str = Field(min_length=2, max_length=50)
    age: int = Field(ge=0, le=120)  # ge = greater equal, le = less equal
    email: str = Field(regex=r'^[^@]+@[^@]+\.[^@]+$')
    score: float = Field(gt=0.0, lt=100.0)  # gt = greater than, lt = less than
```

### 3. Default Values

```python
class Character(BaseModel):
    name: str
    age: int = Field(default=25, description="Character age")
    status: str = Field(default="alive", description="Character status")
    skills: List[str] = Field(default_factory=list)
```

### 4. Advanced Validation

```python
from pydantic import validator, EmailStr
from typing import Literal

class AdvancedUser(BaseModel):
    name: str
    email: EmailStr  # Automatic email validation
    age: int = Field(ge=0, le=120)
    role: Literal["admin", "user", "guest"]  # Enum-like validation
    
    @validator('name')
    def name_must_be_alphanumeric(cls, v):
        assert v.replace(' ', '').isalnum(), 'Name must be alphanumeric'
        return v.title()  # Capitalize each word
```

## Complete Examples

### Movie Review Parser

```python
from pydantic import BaseModel, Field
from typing import List, Literal

class MovieReview(BaseModel):
    title: str = Field(description="Movie title")
    director: str = Field(description="Director name")
    year: int = Field(ge=1888, le=2030, description="Release year")
    genre: List[str] = Field(description="Movie genres")
    rating: float = Field(ge=0.0, le=10.0, description="Rating out of 10")
    sentiment: Literal["positive", "negative", "neutral"]
    pros: List[str] = Field(description="Positive aspects")
    cons: List[str] = Field(description="Negative aspects")
    recommendation: bool = Field(description="Would recommend this movie")

parser = PydanticOutputParser(pydantic_object=MovieReview)

template = PromptTemplate(
    template="""Write a comprehensive review for the movie: {movie}

{format_instructions}

Movie: {movie}""",
    input_variables=["movie"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

chain = template | chat | parser

try:
    result = chain.invoke({"movie": "The Matrix"})
    print(f"Title: {result.title}")
    print(f"Rating: {result.rating}/10")
    print(f"Recommendation: {'Yes' if result.recommendation else 'No'}")
except Exception as e:
    print(f"Parsing error: {e}")
```

### Business Analysis Parser

```python
class BusinessAnalysis(BaseModel):
    company_name: str = Field(description="Company name")
    industry: str = Field(description="Primary industry")
    founded_year: int = Field(ge=1800, le=2024, description="Year founded")
    market_cap: str = Field(description="Market capitalization estimate")
    strengths: List[str] = Field(min_items=2, description="Key strengths")
    weaknesses: List[str] = Field(min_items=1, description="Key weaknesses")
    growth_potential: Literal["high", "medium", "low"]
    investment_rating: float = Field(ge=1.0, le=5.0, description="Investment rating 1-5")
    risk_level: Literal["high", "medium", "low"]

parser = PydanticOutputParser(pydantic_object=BusinessAnalysis)
```

### Scientific Paper Analysis

```python
from datetime import datetime

class PaperAnalysis(BaseModel):
    title: str = Field(description="Paper title")
    authors: List[str] = Field(min_items=1, description="Author names")
    abstract: str = Field(min_length=100, description="Paper abstract")
    field: str = Field(description="Research field")
    methodology: str = Field(description="Research methodology used")
    key_findings: List[str] = Field(min_items=3, description="Main findings")
    significance: Literal["high", "medium", "low"]
    novelty: float = Field(ge=0.0, le=10.0, description="Novelty score out of 10")
    reproducibility: Literal["high", "medium", "low"]
    limitations: List[str] = Field(description="Study limitations")
```

## Error Handling and Validation

### 1. Handling Validation Errors

```python
from pydantic import ValidationError

def safe_parse_character(name):
    try:
        result = chain.invoke({"name": name})
        return result
    except ValidationError as e:
        print("Validation errors:")
        for error in e.errors():
            field = error['loc'][0]
            message = error['msg']
            print(f"- {field}: {message}")
        return None
    except Exception as e:
        print(f"Other error: {e}")
        return None

# Usage
character = safe_parse_character("John Doe")
if character:
    print(f"Created character: {character.name}")
else:
    print("Failed to create character")
```

### 2. Model Validation Details

```python
class StrictUser(BaseModel):
    name: str = Field(min_length=2, description="User name")
    age: int = Field(ge=18, le=100, description="User age")
    email: str = Field(regex=r'^[^@]+@[^@]+\.[^@]+$', description="Valid email")
    
    class Config:
        validate_assignment = True  # Validate when fields are assigned
        extra = "forbid"           # Don't allow extra fields

# This will catch validation errors immediately
try:
    user = StrictUser(name="A", age=15, email="invalid")
except ValidationError as e:
    print("Validation failed:", e)
```

## Advanced Features

### 1. Custom Validators

```python
class AdvancedCharacter(BaseModel):
    name: str
    age: int
    power_level: int = Field(ge=1, le=100)
    
    @validator('name')
    def validate_name(cls, v):
        if len(v.split()) < 2:
            raise ValueError('Name must have at least first and last name')
        return v.title()
    
    @validator('power_level')
    def validate_power_age_relationship(cls, v, values):
        age = values.get('age', 0)
        if age < 18 and v > 50:
            raise ValueError('Young characters cannot have very high power levels')
        return v
```

### 2. Nested Models

```python
class Address(BaseModel):
    street: str
    city: str
    country: str
    postal_code: str = Field(regex=r'^\d{5}(-\d{4})?$')

class ComplexUser(BaseModel):
    name: str
    age: int
    address: Address  # Nested model
    contacts: List[str] = Field(description="Contact methods")
```

### 3. Optional and Union Types

```python
from typing import Union, Optional

class FlexibleData(BaseModel):
    id: Union[int, str]  # Can be either int or string
    value: Optional[float] = None  # Optional field
    metadata: Optional[dict] = None  # Optional dictionary
```

## Best Practices

### 1. Clear Field Descriptions

```python
class WellDocumented(BaseModel):
    name: str = Field(description="Full name of the person")
    age: int = Field(ge=0, le=150, description="Age in years (0-150)")
    email: str = Field(description="Valid email address format")
```

### 2. Appropriate Constraints

```python
class ReasonableConstraints(BaseModel):
    rating: float = Field(ge=1.0, le=5.0, description="Rating from 1.0 to 5.0")
    percentage: float = Field(ge=0.0, le=100.0, description="Percentage 0-100")
    count: int = Field(ge=0, description="Non-negative count")
```

### 3. Default Values for Optional Fields

```python
class RobustModel(BaseModel):
    required_field: str
    optional_with_default: str = Field(default="unknown", description="Optional field")
    optional_list: List[str] = Field(default_factory=list)
    optional_nullable: Optional[str] = None
```

## Model Recommendations

For PydanticOutputParser:
- **Best:** `llama3.1:8b`, `mistral:7b`, `qwen2:7b`
- **Good:** `codellama:7b`, `phi3:medium`
- **Minimum:** Use models with at least 7B parameters
- **Temperature:** 0.1-0.3 for consistent structured output

## When to Use PydanticOutputParser

✅ **Always use for:**
- Production applications
- When data validation is critical
- Complex data structures
- Type safety requirements
- Error handling needs

✅ **Especially good for:**
- APIs requiring consistent data
- Data processing pipelines
- Form validation
- Configuration parsing

## Comparison Summary

| Parser | Validation | Type Safety | Error Handling | Complexity | Production Ready |
|--------|------------|-------------|----------------|------------|------------------|
| Manual | None | None | Manual | Low | No |
| StrOutputParser | None | None | Basic | Low | Limited |
| JsonOutputParser | None | None | Basic | Medium | Limited |
| StructuredOutputParser | Basic | None | Limited | Medium | Partial |
| **PydanticOutputParser** | **Full** | **Yes** | **Excellent** | **High** | **Yes** |

## Exercise

Create a comprehensive news article analyzer with:

```python
class NewsAnalysis(BaseModel):
    headline: str = Field(description="Article headline")
    author: str = Field(description="Article author")
    publication_date: str = Field(description="Publication date")
    word_count: int = Field(ge=100, description="Approximate word count")
    reading_time: int = Field(ge=1, description="Reading time in minutes")
    sentiment: Literal["positive", "negative", "neutral"]
    bias_level: Literal["low", "medium", "high"]
    credibility: float = Field(ge=0.0, le=10.0, description="Credibility score")
    key_topics: List[str] = Field(min_items=2, description="Main topics")
    fact_checkable_claims: List[str] = Field(description="Claims that can be fact-checked")
    political_leaning: Optional[Literal["left", "center", "right"]] = None
```

Test with real news articles and handle validation errors gracefully.

## Conclusion

`PydanticOutputParser` is the **gold standard** for structured output in production applications. It provides:

- **Bulletproof validation** for reliable data
- **Type safety** for better code quality  
- **Excellent error handling** for robust applications
- **Rich feature set** for complex requirements

For any serious LangChain application requiring structured output, PydanticOutputParser should be your first choice.
