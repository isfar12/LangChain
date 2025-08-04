# Tutorial 1: Without Output Parser

This tutorial demonstrates the manual approach to handling LLM outputs without using any output parsers.

## Learning Objectives

- Understand the baseline approach to LLM output handling
- See the challenges of manual processing
- Learn why output parsers are needed

## Code Explanation

```python
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama

model = ChatOllama(
    model="gemma:2b",
    temperature=0.5,
)

# Two separate prompt templates
template1 = PromptTemplate(
    template="Write me a detailed report on {topic}",
    input_variables=["topic"],
)

template2 = PromptTemplate(
    template="Summarize the key points of {topic}",
    input_variables=["topic"],
)

# Manual step-by-step process
prompt1 = template1.invoke({"topic": "Artificial Intelligence"})
result = model.invoke(prompt1)

# Manually extract content from first result
prompt2 = template2.invoke({"topic": result.content})
result2 = model.invoke(prompt2)

print(result.content)
```

## Step-by-Step Breakdown

### 1. Model Setup

```python
model = ChatOllama(model="gemma:2b", temperature=0.5)
```

- Creates an Ollama chat model instance
- Temperature 0.5 for balanced creativity/consistency

### 2. Template Definition

```python
template1 = PromptTemplate(
    template="Write me a detailed report on {topic}",
    input_variables=["topic"],
)
```

- Defines a reusable prompt template
- `{topic}` is a placeholder for dynamic content

### 3. Manual Processing

```python
prompt1 = template1.invoke({"topic": "Artificial Intelligence"})
result = model.invoke(prompt1)
```

- Manually invoke each step
- Extract `.content` manually from model response

### 4. Chaining Manually

```python
prompt2 = template2.invoke({"topic": result.content})
result2 = model.invoke(prompt2)
```

- Use output from first model as input to second
- Requires manual content extraction

## Challenges with Manual Approach

### 1. **Verbose Code**

```python
# Manual approach - lots of steps
prompt1 = template1.invoke({"topic": "AI"})
result1 = model.invoke(prompt1)
content1 = result1.content  # Manual extraction

prompt2 = template2.invoke({"topic": content1})
result2 = model.invoke(prompt2)
content2 = result2.content  # Manual extraction again
```

### 2. **Error Prone**

- Must remember to extract `.content` each time
- Easy to forget or make mistakes
- No automatic error handling

### 3. **Hard to Chain**

- Each step requires manual intervention
- Difficult to create complex workflows
- No automatic data flow

### 4. **No Data Validation**

- No way to ensure output format
- Raw text only - no structured data
- Must manually parse any structure

## Improved Approach Preview

With output parsers, the same workflow becomes:

```python
from langchain_core.output_parsers import StrOutputParser

parser = StrOutputParser()

# Clean, chainable approach
chain = template1 | model | parser | template2 | model | parser
result = chain.invoke({"topic": "Artificial Intelligence"})
```

## When to Use Manual Approach

✅ **Good for:**

- Simple, one-off tasks
- Learning LangChain basics
- Quick prototyping
- Single-step operations

❌ **Avoid for:**

- Complex workflows
- Production applications
- Chained operations
- Structured data needs

## Exercise

Try modifying the code to:

1. **Add a third step** - Ask for specific recommendations
2. **Handle errors** - What happens if the model fails?
3. **Extract specific information** - Parse out key points manually

## Key Takeaways

- Manual processing requires extracting `.content` from each model response
- This approach is verbose and error-prone
- Output parsers solve these problems by automating data extraction
- Understanding the manual approach helps appreciate parser benefits

## Next Step

Move to Tutorial 2 (`stroutputparser.py`) to see how `StrOutputParser` simplifies this workflow.
