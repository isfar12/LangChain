# Output Parsers in LangChain

This folder demonstrates different output parsing techniques in LangChain to process and structure LLM responses.

## What are Output Parsers?

Output parsers are tools that convert raw LLM text responses into structured, usable data formats. They help ensure consistent, predictable output from language models.

## Files Overview

### 1. `without_stroutputparser.py`
Manual approach without parsers - shows the baseline method of handling LLM outputs.

### 2. `stroutputparser.py`
Using `StrOutputParser` for simple string processing and chaining operations.

### 3. `jsonoutputparser.py`
Using `JsonOutputParser` for basic JSON output without validation.

### 4. `structured_output_using_parsers.py`
Using `StructuredOutputParser` with `ResponseSchema` for defined structure.

### 5. `structured_output_using_pydantic.py`
Using `PydanticOutputParser` for advanced validation and type checking.

## Progression of Complexity

```
Manual Processing → String Parser → JSON Parser → Structured Parser → Pydantic Parser
     (Basic)         (Chaining)     (JSON)       (Schema)        (Validation)
```

## When to Use Each Parser

| Parser Type | Use Case | Validation | Complexity |
|------------|----------|------------|------------|
| **Manual** | Simple responses | None | Low |
| **StrOutputParser** | Text processing & chaining | None | Low |
| **JsonOutputParser** | Basic JSON output | None | Medium |
| **StructuredOutputParser** | Defined schema | Basic | Medium |
| **PydanticOutputParser** | Type validation | Advanced | High |

## Best Practices

1. **Start Simple** - Use `StrOutputParser` for basic text processing
2. **Use Pydantic for Production** - Best validation and error handling
3. **Lower Temperature** - Use 0.1-0.3 for structured output
4. **Clear Instructions** - Be explicit about expected format
5. **Handle Errors** - Always wrap parsing in try-catch blocks

## Model Recommendations

For output parsing with Ollama:
- **Best**: `llama3.1:8b`, `mistral:7b`
- **Good**: `qwen2:7b`, `codellama:7b`
- **Avoid**: Very small models (< 7B parameters)

## Common Issues

- **Parsing Failures**: Model returns unexpected format
- **Validation Errors**: Data doesn't match expected schema
- **Empty Output**: Model doesn't follow instructions

**Solutions**: Use better models, clearer prompts, lower temperature, and proper error handling.
