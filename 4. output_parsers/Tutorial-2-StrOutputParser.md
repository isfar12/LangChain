# Tutorial 2: StrOutputParser

This tutorial demonstrates using `StrOutputParser` for simple string processing and creating smooth LLM chains.

## Learning Objectives

- Understand how `StrOutputParser` simplifies output handling
- Learn to create seamless LLM chains
- See the benefits of automated content extraction

## Code Explanation

```python
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama

model = ChatOllama(
    model="gemma:2b",
    temperature=0.5,
)

template1 = PromptTemplate(
    template="Write me a detailed report on {topic}",
    input_variables=["topic"],
)

template2 = PromptTemplate(
    template="Summarize the key points of {topic}",
    input_variables=["topic"],
)

# StrOutputParser automatically extracts content
parser = StrOutputParser()

# Clean chain using pipe operator
chain = template1 | model | parser | template2 | model | parser

result = chain.invoke({"topic": "Artificial Intelligence"})
print(result)
```

## What is StrOutputParser?

`StrOutputParser` is the simplest output parser that:
- **Automatically extracts** `.content` from model responses
- **Returns plain strings** for easy chaining
- **Handles errors** gracefully
- **Enables clean syntax** with pipe operators

## Key Improvements Over Manual Approach

### 1. Automatic Content Extraction

**Manual Approach:**
```python
result = model.invoke(prompt)
content = result.content  # Manual extraction
```

**With StrOutputParser:**
```python
parser = StrOutputParser()
content = (model | parser).invoke(prompt)  # Automatic
```

### 2. Clean Chaining Syntax

**Manual Approach:**
```python
prompt1 = template1.invoke({"topic": "AI"})
result1 = model.invoke(prompt1)
content1 = result1.content

prompt2 = template2.invoke({"topic": content1})
result2 = model.invoke(prompt2)
content2 = result2.content
```

**With StrOutputParser:**
```python
chain = template1 | model | parser | template2 | model | parser
result = chain.invoke({"topic": "AI"})
```

### 3. Error Handling

StrOutputParser handles various model response formats:
- Standard chat responses
- Empty responses
- Error responses

## Understanding the Chain

Let's break down this chain step by step:

```python
chain = template1 | model | parser | template2 | model | parser
```

### Step-by-Step Flow:

1. **`template1`** - Creates prompt from input `{"topic": "AI"}`
2. **`model`** - Generates detailed report
3. **`parser`** - Extracts string content automatically
4. **`template2`** - Uses report content to create summary prompt
5. **`model`** - Generates summary
6. **`parser`** - Extracts final string result

### Data Flow Visualization:

```
Input: {"topic": "AI"}
    ↓
template1: "Write me a detailed report on AI"
    ↓
model: AIMessage(content="AI is a field...")
    ↓
parser: "AI is a field..."  ← Automatic .content extraction
    ↓
template2: "Summarize the key points of AI is a field..."
    ↓
model: AIMessage(content="Key points: 1. Machine learning...")
    ↓
parser: "Key points: 1. Machine learning..."  ← Final result
```

## Practical Examples

### Simple Question-Answer Chain
```python
from langchain_core.output_parsers import StrOutputParser

parser = StrOutputParser()

question_template = PromptTemplate(
    template="What is {concept}?",
    input_variables=["concept"]
)

explanation_template = PromptTemplate(
    template="Explain this in simple terms: {explanation}",
    input_variables=["explanation"]
)

chain = question_template | model | parser | explanation_template | model | parser

result = chain.invoke({"concept": "quantum computing"})
print(result)
```

### Multi-Step Analysis Chain
```python
analyze_template = PromptTemplate(
    template="Analyze the pros and cons of {topic}",
    input_variables=["topic"]
)

conclusion_template = PromptTemplate(
    template="Based on this analysis, what's your recommendation? {analysis}",
    input_variables=["analysis"]
)

chain = analyze_template | model | parser | conclusion_template | model | parser

result = chain.invoke({"topic": "remote work"})
print(result)
```

## Benefits of StrOutputParser

✅ **Automatic Processing** - No manual `.content` extraction
✅ **Clean Syntax** - Readable pipe operator chains
✅ **Error Handling** - Graceful handling of edge cases
✅ **Type Safety** - Always returns strings
✅ **Composable** - Easy to combine with other components

## Limitations

❌ **No Validation** - Doesn't check output format
❌ **String Only** - Can't parse structured data
❌ **No Type Checking** - No schema enforcement
❌ **Basic Functionality** - Limited to simple text processing

## When to Use StrOutputParser

✅ **Good for:**
- Simple text processing chains
- When you need clean string output
- Chaining multiple LLM calls
- Prototyping workflows

❌ **Consider alternatives for:**
- Structured data extraction
- JSON parsing needs
- Data validation requirements
- Complex data transformation

## Common Patterns

### Research Chain
```python
research_chain = (
    research_template | model | parser |
    analysis_template | model | parser |
    conclusion_template | model | parser
)
```

### Translation Chain
```python
translation_chain = (
    translate_template | model | parser |
    improve_template | model | parser
)
```

### Writing Assistant Chain
```python
writing_chain = (
    draft_template | model | parser |
    edit_template | model | parser |
    polish_template | model | parser
)
```

## Best Practices

1. **Use consistent naming** for templates and chains
2. **Keep chains readable** - break long chains into steps
3. **Add error handling** around chain invocation
4. **Test with different inputs** to ensure robustness

## Troubleshooting

**Problem:** Chain fails with parsing error
**Solution:** Check that each step returns expected format

**Problem:** Unexpected output format
**Solution:** Add debugging prints between chain steps

**Problem:** Empty results
**Solution:** Verify model and template work individually

## Exercise

Create a chain that:
1. Takes a movie title as input
2. Generates a plot summary
3. Analyzes the genre
4. Provides a rating recommendation

## Next Step

Move to Tutorial 3 (`jsonoutputparser.py`) to learn about parsing JSON output from LLMs.
