# Structured Output with LangChain - Complete Tutorial

This comprehensive guide demonstrates three different approaches to getting structured output from LLMs using LangChain and Ollama models. Learn when and how to use each approach for reliable, consistent data extraction from AI models.

## 🎯 Why Structured Output?

Structured output is crucial for building reliable AI applications. Instead of parsing unpredictable free-form text responses, you can enforce specific data formats that your application can process with confidence.

**Benefits:**
- ✅ **Predictable data format** for downstream processing
- ✅ **Type safety** and validation
- ✅ **Automatic parsing** - no manual text processing needed
- ✅ **Error handling** for malformed responses
- ✅ **Integration-ready** data structures

## 📁 Directory Structure

```
3.structured_output/
├── 1.typedict/                    # Basic type hints approach
│   ├── typedict_basic.py          # Standalone TypedDict demo
│   ├── struct_out_using_typedict.py # LangChain integration
│   └── README.md                  # TypedDict documentation
├── 2.pydantic/                    # Advanced validation approach ⭐ RECOMMENDED
│   ├── pydantic_basic.py          # Standalone Pydantic demo  
│   ├── struct_out_using_pydantic.py # LangChain integration
│   └── README.md                  # Pydantic documentation
├── 3.json/                        # Maximum flexibility approach
│   ├── basic_json.json            # Example JSON schema
│   ├── struct_using_json.py       # LangChain integration
│   └── README.md                  # JSON Schema documentation
└── Readme.md                      # This comprehensive guide
```

## 🚀 Quick Start Recommendations

### **For Beginners**: Start with TypedDict (`1.typedict/`)
### **For Production**: Use Pydantic (`2.pydantic/`) ⭐ **RECOMMENDED**
### **For Complex Schemas**: Consider JSON Schema (`3.json/`)

---

## 1️⃣ **TypedDict Approach** - Simple Type Hints

### **What is TypedDict?**

TypedDict provides basic type hints for dictionaries with fixed keys. It's the simplest approach - think of it as "typed dictionaries" with no runtime validation.

### **Basic Usage Example**

```python
from typing import TypedDict

class Person(TypedDict):
    name: str
    age: int
    is_student: bool

# Create instance (just a regular dictionary)
person: Person = {
    "name": "Alice", 
    "age": 30, 
    "is_student": False
}
```

### **LangChain Integration**

```python
from langchain_ollama import ChatOllama
from typing import TypedDict

class UserProfile(TypedDict):
    name: str
    email: str
    age: int
    interests: list[str]

model = ChatOllama(model="llama3.1:8b", temperature=0.1)
structured_llm = model.with_structured_output(UserProfile)

result = structured_llm.invoke(
    "Extract user info: John Doe, john@email.com, 25 years old, likes programming and hiking"
)

print(result)  # Returns: {'name': 'John Doe', 'email': 'john@email.com', ...}
```

### **TypedDict Features**

✅ **Pros:**
- Simple and lightweight
- Good IDE autocomplete support
- Minimal learning curve
- No additional dependencies

❌ **Cons:**
- No runtime validation
- No default values
- No field constraints
- Basic types only

### **When to Use TypedDict**
- ✅ Simple prototyping and experimentation
- ✅ Basic dictionary structures
- ✅ When you don't need validation
- ❌ Not recommended for production applications

---

## 2️⃣ **Pydantic Approach** - Advanced Validation ⭐ **RECOMMENDED**

### **What is Pydantic?**

Pydantic is a powerful data validation library that provides automatic type checking, data conversion, and validation at runtime. It's the **recommended approach** for most applications.

### **Basic Usage Example**

```python
from pydantic import BaseModel, EmailStr, Field
from typing import Optional

class User(BaseModel):
    name: str = "unknown"  # Default value
    email: EmailStr        # Automatic email validation
    age: Optional[int] = Field(None, ge=0, description="Must be non-negative")

# Create with automatic validation
user = User(name="John", email="john@example.com", age=25)
print(user.dict())  # Convert to dictionary
```

### **Advanced Features Demo**

```python
from pydantic import BaseModel, Field
from typing import List, Literal, Annotated

class ProductReview(BaseModel):
    title: Annotated[str, Field(description="Review title")]
    rating: int = Field(ge=1, le=5, description="Rating from 1-5 stars")
    sentiment: Literal["positive", "neutral", "negative"]  # Only these values allowed
    tags: List[str] = Field(default_factory=list)  # Empty list if not provided
    verified: bool = True

# Automatic validation - will raise error if invalid
review = ProductReview(
    title="Great product!",
    rating=5,  # Must be 1-5
    sentiment="positive",  # Must be one of the literals
    tags=["electronics", "recommended"]
)
```

### **LangChain Integration**

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

print(f"Title: {result.title}")      # Direct attribute access
print(f"Genre: {result.genre}")      # Type-safe and validated
print(f"Rating: {result.rating}")
```

### **Key Pydantic Features**

#### **1. Automatic Validation**
```python
age: int = Field(ge=0, le=120)  # Must be between 0 and 120
email: EmailStr                 # Must be valid email format
password: str = Field(min_length=8)  # Minimum 8 characters
```

#### **2. Default Values & Optional Fields**
```python
name: str = "Anonymous"
active: bool = True
tags: List[str] = Field(default_factory=list)  # Empty list default
metadata: Optional[Dict[str, Any]] = None      # Can be None
```

#### **3. Rich Type System**
```python
status: Literal["active", "inactive", "pending"]  # Enum-like values
scores: List[float] = Field(description="List of test scores")
config: Dict[str, Union[str, int]] = {}
```

#### **4. Custom Validation**
```python
from pydantic import validator

class User(BaseModel):
    username: str
    
    @validator('username')
    def username_must_be_alphanumeric(cls, v):
        assert v.isalnum(), 'Username must be alphanumeric'
        return v
```

### **Pydantic Benefits Over TypedDict**

✅ **Runtime validation** - Catches errors when data is created
✅ **Default values** - Automatic handling of optional fields  
✅ **Field constraints** - Min/max values, string patterns, etc.
✅ **Type conversion** - Automatic conversion between compatible types
✅ **Better error messages** - Clear validation error descriptions
✅ **Serialization** - Easy conversion to/from JSON, dict, etc.

### **Common Pydantic Patterns**

#### **User Profile Extraction**
```python
class UserProfile(BaseModel):
    name: str = Field(description="Full name")
    email: EmailStr = Field(description="Email address")
    age: Optional[int] = Field(None, ge=0, le=120)
    skills: List[str] = Field(default_factory=list)
```

#### **Sentiment Analysis**
```python
class SentimentAnalysis(BaseModel):
    text: str = Field(description="Original text")
    sentiment: Literal["positive", "negative", "neutral"]
    confidence: float = Field(ge=0.0, le=1.0)
    keywords: List[str] = Field(default_factory=list)
```

#### **Data Extraction**
```python
class Invoice(BaseModel):
    invoice_number: str
    date: str = Field(description="Invoice date in YYYY-MM-DD format")
    total: float = Field(ge=0, description="Total amount")
    items: List[str] = Field(description="List of items/services")
```

---

## 3️⃣ **JSON Schema Approach** - Maximum Control

### **What is JSON Schema?**

JSON Schema provides maximum flexibility for defining data structures with complex validation rules. It's an industry standard that gives you complete control over validation logic.

### **Basic JSON Schema Structure**

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

### **LangChain Integration**

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
    "additionalProperties": False  # Don't allow extra fields
}

model = ChatOllama(model="llama3.1:8b", temperature=0.1)
structured_llm = model.with_structured_output(schema)

result = structured_llm.invoke(
    "Extract info: John Doe, 30 years old, john@email.com, knows Python and JavaScript"
)

print(json.dumps(result, indent=2))
```

### **JSON Schema Components**

#### **1. Basic Types**
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

#### **2. String Validation**
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

#### **3. Number Constraints**
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

#### **4. Array Definitions**
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

#### **5. Enum Values**
```json
{
  "status": {
    "type": "string",
    "enum": ["active", "inactive", "pending"]
  }
}
```

### **Complex Example: E-commerce Product**

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

model = ChatOllama(model="llama3.1:8b", temperature=0.1)
structured_llm = model.with_structured_output(ecommerce_schema)
```

### **JSON Schema Advantages**

✅ **Maximum flexibility** - Define any validation rule possible
✅ **Industry standard** - Widely supported format
✅ **Rich validation** - Complex constraints and conditional logic
✅ **Self-documenting** - Schemas serve as documentation
✅ **Tool ecosystem** - Many tools support JSON Schema

### **JSON Schema Limitations**

❌ **Verbose** - More code than Pydantic for simple cases
❌ **Python dictionary required** - Must convert JSON to dict for LangChain
❌ **Limited IDE support** - Less autocomplete compared to Pydantic
❌ **Learning curve** - Complex syntax for advanced features

---

## ⚖️ **Comparison: When to Use Each Approach**

| Feature | TypedDict | Pydantic | JSON Schema |
|---------|-----------|----------|-------------|
| **Complexity** | Simple | Medium | High |
| **Runtime Validation** | ❌ | ✅ | ✅ |
| **Default Values** | ❌ | ✅ | ✅ |
| **Field Constraints** | ❌ | ✅ | ✅ |
| **IDE Support** | ✅ | ✅ | ❌ |
| **Learning Curve** | Easy | Medium | Steep |
| **Best For** | Prototyping | Production | Complex validation |

### **Decision Matrix**

#### **Choose TypedDict if:**
- ✅ Simple prototyping
- ✅ Basic type hints needed
- ✅ No validation required
- ✅ Minimal dependencies preferred

#### **Choose Pydantic if:** ⭐ **RECOMMENDED**
- ✅ Production applications
- ✅ Need runtime validation
- ✅ Want great developer experience
- ✅ Building Python-centric apps
- ✅ Need good balance of features and simplicity

#### **Choose JSON Schema if:**
- ✅ Complex validation requirements
- ✅ Need maximum control over schema
- ✅ Integration with non-Python systems
- ✅ API documentation generation needed

---

## 🛠️ **Best Practices for All Approaches**

### **1. Model Configuration**
```python
# Use consistent, appropriate model settings
model = ChatOllama(
    model="llama3.1:8b",  # Best for structured output
    temperature=0.1       # Low temperature for consistency
)
```

### **2. Descriptive Field Documentation**
```python
# Good: Helps LLM understand what to extract
name: str = Field(description="Full name of the person")

# Bad: No guidance for the LLM
name: str
```

### **3. Sensible Defaults and Optional Fields**
```python
# Make fields optional when they might not be available
age: Optional[int] = Field(None, ge=0, description="Age if mentioned")
tags: List[str] = Field(default_factory=list, description="Relevant tags")
```

### **4. Clear Prompts**
```python
# Good: Explicit about expected structure
prompt = """
Extract the following information from this text:
- Full name
- Email address 
- Age (if mentioned)
- Skills/interests as a list

Text: "John Doe is 25 years old, email john@example.com, knows Python and ML"
"""

# Bad: Vague request
prompt = "Extract info from this text"
```

---

## 🚀 **Model Recommendations**

### **Best Models for Structured Output:**
1. **`llama3.1:8b`** - Best overall performance ⭐
2. **`mistral:7b`** - Good alternative 
3. **`qwen2:7b`** - Excellent instruction following
4. **`codellama:7b`** - Good for JSON-like structures

### **Model Settings:**
```python
# Recommended configuration
model = ChatOllama(
    model="llama3.1:8b",
    temperature=0.1,      # Low for consistency
    # top_p=0.9,          # Optional: nucleus sampling
    # repeat_penalty=1.1   # Optional: reduce repetition
)
```

---

## 🔧 **Troubleshooting Common Issues**

### **Problem: Empty or malformed output**
**Solution:** 
- Use more explicit prompts
- Lower the temperature (0.1-0.3)
- Try different models (llama3.1:8b works best)

### **Problem: Missing required fields**
**Solution:**
- Make fields optional with defaults
- Be more explicit in field descriptions
- Check if the information exists in the input text

### **Problem: Type validation errors (Pydantic)**
**Solution:**
- Use `Optional[]` for fields that might be missing
- Add field constraints that match expected data
- Provide sensible default values

### **Problem: Schema too complex (JSON Schema)**
**Solution:**
- Break complex schemas into smaller, nested objects
- Start simple and add complexity gradually
- Consider switching to Pydantic for better developer experience

---

## 📚 **Learning Path**

### **Step 1: Start Simple** (30 minutes)
- Run `1.typedict/typedict_basic.py`
- Try `1.typedict/struct_out_using_typedict.py`
- Understand basic concept of structured output

### **Step 2: Level Up** (1 hour) ⭐ **RECOMMENDED**
- Study `2.pydantic/pydantic_basic.py`
- Experiment with `2.pydantic/struct_out_using_pydantic.py`
- Build your own Pydantic models

### **Step 3: Advanced Control** (2 hours)
- Explore `3.json/basic_json.json`
- Try `3.json/struct_using_json.py`
- Compare with Pydantic approach

### **Step 4: Real Projects**
- Build a data extraction system
- Create a content analysis tool
- Develop a form parser application

---

## 🎯 **Next Steps**

After mastering structured output, you'll be ready for:
- **RAG applications** with structured document parsing
- **Agent systems** that use structured tool inputs/outputs  
- **Data pipeline automation** with reliable extraction
- **API integration** with validated data structures
- **Production applications** with robust error handling

## 🔗 **Related Resources**

- [LangChain Structured Output Documentation](https://python.langchain.com/docs/modules/model_io/output_parsers/structured)
- [Pydantic Documentation](https://docs.pydantic.dev/)
- [JSON Schema Specification](https://json-schema.org/)
- [TypedDict Documentation](https://docs.python.org/3/library/typing.html#typing.TypedDict)

---

**Remember:** Start with Pydantic for most use cases - it provides the best balance of features, validation, and developer experience! 🎉
- **Inconsistent format**: Switch to a more reliable model like llama3.1*clear explanation** of **Structured Output in LangChain** with **simple analogies** to make it easy to grasp:

---

## 🌟 **What is Structured Output in LangChain?**

When you ask an AI a question, it usually replies with **free text** (like chatting with a friend).
But sometimes, you **don’t want just words** — you need the answer in a **specific, organized format** (like a JSON, table, or a checklist).

✅ **Structured Output** in LangChain means:

- Forcing the AI to reply in a **fixed format** (e.g., JSON, dictionary, list of objects).
- Making it easier for software to **read, store, or process** the response.

📌 **Analogy:**
Imagine you’re a chef (AI) taking orders from a waiter (LangChain).

- If you just “talk,” you might say: *“I’ll make a pizza with cheese, some olives, maybe mushrooms.”*
- But the restaurant POS system (your app) needs the order in a **structured format**:

```json
{
  "dish": "pizza",
  "toppings": ["cheese", "olives", "mushrooms"]
}
```

This way, the **kitchen staff and billing system** know exactly what to do.

---

## 🔧 **How Does LangChain Help?**

LangChain provides **tools** to make sure the AI **sticks to the structure**:

- **Output Parsers** → These tell the AI how the final response should “look.”
- **Schemas** → Define the “template” for answers (e.g., JSON with specific fields).
- **Validation** → Checks if the AI’s answer actually fits the structure.

📌 **Analogy:**
Think of LangChain like a **template in Google Forms**.

- The form **forces people to fill boxes** like Name, Email, and Age.
- No matter how they feel like writing, they must follow the **boxes you gave**.

---

## 🤖 **Why is This Important for Generative AI?**

Without structured output, AI might say:

> “The weather is sunny with a slight breeze, and the temperature feels like 26°C.”

But if your app expects **structured weather data**, that’s messy.

With structured output:

```json
{
  "temperature": 26,
  "condition": "sunny",
  "wind": "slight breeze"
}
```

✅ Easier for apps to:

- **Store in databases**
- **Use in dashboards**
- **Trigger other actions** (e.g., send an umbrella reminder if “rainy”)

📌 **Analogy:**
Like getting your **salary** — you don’t want your boss to hand you *“some money”*.
You want a **salary slip**: how much is **basic pay**, how much is **bonus**, how much is **tax**. That’s **structured**!

---

## 🏗 **How You Use It in LangChain**

1️⃣ **Define a schema** – tell LangChain what fields you want (e.g., name, age, city).
2️⃣ **Tell the AI** – “always reply in this structure.”
3️⃣ **Use an Output Parser** – LangChain checks the AI’s reply and “fixes” it if needed.

📌 **Example:**

```python
from langchain.output_parsers import StructuredOutputParser
from langchain.prompts import ChatPromptTemplate

# Define schema
schema = {
  "name": "string",
  "age": "integer",
  "city": "string"
}
```

The AI must **follow this schema** — like filling out a **government form**.

---

## 🎯 **Simple Takeaway**

- **Without structured output**: AI replies like a chatty friend.
- **With structured output**: AI replies like a **Google Sheet row** — neat, tidy, and ready for use.

📌 **Analogy:**
Think of **structured output** as putting AI answers into **labeled boxes** instead of dumping them into a bag. It’s easier to **find, use, and trust** the information.

---

Would you like me to:
✅ **Show a real LangChain code example** with structured output (JSON)?
✅ Or **make a diagram** showing “AI → LangChain → Structured Output”?
