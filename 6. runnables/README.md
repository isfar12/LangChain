# Runnables in LangChain

## Table of Contents

1. [Introduction to Runnables](#introduction-to-runnables)
2. [Prerequisites](#prerequisites)
3. [Tutorial 1: Runnable Sequence](#tutorial-1-runnable-sequence)
4. [Tutorial 2: Runnable Parallel](#tutorial-2-runnable-parallel)
5. [Tutorial 3: Runnable Passthrough](#tutorial-3-runnable-passthrough)
6. [Tutorial 4: Runnable Lambda](#tutorial-4-runnable-lambda)
7. [Tutorial 5: Runnable Branch](#tutorial-5-runnable-branch)
8. [Running the Examples](#running-the-examples)
9. [Key Concepts Summary](#key-concepts-summary)

## Introduction to Runnables

Runnables are the fundamental building blocks of LangChain that allow you to create sophisticated AI workflows. They provide a unified interface for executing different operations in sequence, parallel, or with conditional logic. Think of Runnables as composable components that can be chained together to create complex AI pipelines.

### What are Runnables?

- **Composable**: Chain multiple operations together
- **Reusable**: Create modular components that can be used across different workflows
- **Flexible**: Support sequential, parallel, and conditional execution
- **Standardized**: All Runnables follow the same interface with `.invoke()`, `.stream()`, and `.batch()` methods

## Prerequisites

Before running these examples, ensure you have:

1. **Ollama installed** with the following models:
   - `gemma3:1b`
   - `gemma2:2b`
   - `llama3.1`

2. **Required Python packages**:

   ```bash
   pip install langchain-ollama langchain-core
   ```

3. **Ollama models downloaded**:

   ```bash
   ollama pull gemma3:1b
   ollama pull gemma2:2b
   ollama pull llama3.1
   ```

## Tutorial 1: Runnable Sequence

**File**: `1. runnable_sequence.py`

### Concept

`RunnableSequence` executes multiple operations one after another in a linear fashion. The output of one operation becomes the input of the next.

### Code Explanation

```python
from langchain_core.runnables import RunnableSequence
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
```

**Imports**:

- `RunnableSequence`: Core class for sequential execution
- `ChatOllama`: Interface to Ollama chat models
- `PromptTemplate`: Template for creating structured prompts
- `StrOutputParser`: Converts model output to string

```python
model = ChatOllama(
    model="gemma3:1b",
    temperature=0.7
)
```

**Model Setup**:

- Uses Gemma 3 1B model via Ollama
- Temperature 0.7 for balanced creativity and consistency

```python
prompt1 = PromptTemplate(
    template="Write me a joke about {topic}",
    input_variables=["topic"]
)

prompt2 = PromptTemplate(
    template="Explain the following joke: {joke} \n First write the joke, then explain it.",
    input_variables=["joke"]
)
```

**Prompt Templates**:

- `prompt1`: Generates a joke about a given topic
- `prompt2`: Explains the generated joke

```python
chain = RunnableSequence(prompt1, model, parser, prompt2, model, parser)
```

**Chain Construction**:
The sequence flows as follows:

1. `prompt1` → Creates joke prompt
2. `model` → Generates joke
3. `parser` → Converts to string
4. `prompt2` → Creates explanation prompt using the joke
5. `model` → Generates explanation
6. `parser` → Converts final output to string

### Execution Flow

```bash
Input: {"topic": "cats"}
↓
prompt1: "Write me a joke about cats"
↓
model: Generates joke
↓
parser: Converts to string
↓
prompt2: "Explain the following joke: [generated joke]..."
↓
model: Generates explanation
↓
parser: Final string output
```

### How to Run

```bash
cd "e:\LangChain\6. runnables"
python "1. runnable_sequence.py"
```

**Expected Output**: A joke about cats followed by an explanation of why it's funny.

---

## Tutorial 2: Runnable Parallel

**File**: `2. runnable_parellal.py`

### Concept

`RunnableParallel` executes multiple operations simultaneously, each receiving the same input. This is useful when you want to generate different types of content from the same input.

### Code Explanation

```python
model1 = ChatOllama(model="gemma3:1b", temperature=0.7)
model2 = ChatOllama(model="gemma2:2b", temperature=0.7)
```

**Multiple Models**:

- Uses two different models for variety in outputs
- Each model may have different strengths or characteristics

```python
prompt1 = PromptTemplate(
    template="Write me a Linkedin medium length post about {topic}, Just give me the post, no explanation.",
    input_variables=["topic"]
)

prompt2 = PromptTemplate(
    template="Write me a tweet about {topic}, just give me the tweet, no explanation.",
    input_variables=["topic"]
)
```

**Different Content Types**:

- `prompt1`: Creates LinkedIn posts (longer form)
- `prompt2`: Creates tweets (short form)

```python
parallel_chain = RunnableParallel({
    "linkedin_post": RunnableSequence(prompt1, model1, parser),
    "tweet": RunnableSequence(prompt2, model2, parser)
})
```

**Parallel Execution**:

- Both sequences run simultaneously
- Results are collected in a dictionary with specified keys
- Each branch is independent and can use different models/prompts

### Execution Flow

```bash
Input: {"topic": "AI and its impact on society"}
↓
┌─────────────────────────────────────┬─────────────────────────────────────┐
│ "linkedin_post" branch              │ "tweet" branch                      │
│ prompt1 → model1 → parser           │ prompt2 → model2 → parser           │
└─────────────────────────────────────┴─────────────────────────────────────┘
↓
Output: {"linkedin_post": "...", "tweet": "..."}
```

### How to Run

```bash
python "2. runnable_parellal.py"
```

**Expected Output**: A dictionary containing both a LinkedIn post and a tweet about the same topic.

---

## Tutorial 3: Runnable Passthrough

**File**: `3. runnable_passthrough.py`

### Concept

`RunnablePassthrough` forwards its input unchanged. This is useful when you want to preserve original data while also processing it in parallel branches.

### Code Explanation

```python
linkedin_generator = RunnableSequence(prompt1, model1, parser)
```

**Content Generator**:

- Creates the initial LinkedIn post
- This becomes the input for the parallel processing

```python
parallel_chain = RunnableParallel({
    "linkedin_post": RunnablePassthrough(),
    "summary": RunnableSequence(prompt2, model2, parser)
})
```

**Passthrough Usage**:

- `"linkedin_post"`: Uses `RunnablePassthrough()` to preserve the original generated post
- `"summary"`: Processes the post to create a summary

```python
final_chain = RunnableSequence(linkedin_generator, parallel_chain)
```

**Combined Chain**:

1. Generate LinkedIn post
2. Pass it to parallel processing
3. Return both original and processed versions

### Execution Flow

```bash
Input: {"topic": "AI and its impact on society"}
↓
linkedin_generator: Generates LinkedIn post
↓
parallel_chain:
┌─────────────────────────────────────┬─────────────────────────────────────┐
│ "linkedin_post"                     │ "summary"                           │
│ RunnablePassthrough()               │ prompt2 → model2 → parser           │
│ (preserves original post)           │ (creates summary)                   │
└─────────────────────────────────────┴─────────────────────────────────────┘
↓
Output: {"linkedin_post": "original post", "summary": "summarized post"}
```

### How to Run

```bash
python "3. runnable_passthrough.py"
```

**Expected Output**: A dictionary with the original LinkedIn post and its summary.

---

## Tutorial 4: Runnable Lambda

**File**: `4. runnable_lambda.py`

### Concept

`RunnableLambda` allows you to create custom functions that can be integrated into Runnable chains. This enables custom processing, transformations, or calculations.

### Code Explanation

```python
def count_words(text):
    return len(text.split())
```

**Custom Function**:

- Simple function that counts words in text
- Can be any Python function that takes input and returns output

```python
linkedin_generator = RunnableSequence(prompt1, chat, parser)

parallel_chain = RunnableParallel({
    "linkedin_post": linkedin_generator,
    "word_count": RunnableLambda(count_words)
})

final_chain = RunnableSequence(linkedin_generator, parallel_chain)
```

**Lambda Integration**:

- `RunnableLambda(count_words)` wraps the custom function
- The function receives the LinkedIn post as input
- Returns the word count

### Execution Flow

```
Input: {"topic": "AI and its impact on society"}
↓
linkedin_generator: Generates LinkedIn post
↓
parallel_chain:
┌─────────────────────────────────────┬─────────────────────────────────────┐
│ "linkedin_post"                     │ "word_count"                        │
│ linkedin_generator                  │ RunnableLambda(count_words)         │
│ (generates post)                    │ (counts words in post)              │
└─────────────────────────────────────┴─────────────────────────────────────┘
↓
Output: {"linkedin_post": "post content", "word_count": 150}
```

### Common Use Cases for RunnableLambda

- Data transformation and cleaning
- Custom calculations (word count, readability scores)
- API calls to external services
- File operations
- Custom validation logic

### How to Run

```bash
python "4. runnable_lambda.py"
```

**Expected Output**: A dictionary with the LinkedIn post and its word count.

---

## Tutorial 5: Runnable Branch

**File**: `5. runnable_branch.py`

### Concept

`RunnableBranch` provides conditional execution based on the input or intermediate results. It evaluates conditions and routes execution to different branches accordingly.

### Code Explanation

```python
model = ChatOllama(model="llama3.1", temperature=0.5)
```

**Model Setup**:

- Uses Llama 3.1 for consistent, focused output
- Lower temperature (0.5) for more deterministic results

```python
prompt = PromptTemplate(
    template="Write a report on the following topic: {topic}",
    input_variables=["topic"],
)

summarize_prompt = PromptTemplate(
    template="Summarize the following report within 150 words: {text}",
    input_variables=["text"],
)
```

**Conditional Prompts**:

- `prompt`: Generates initial report
- `summarize_prompt`: Used only if report is too long

```python
report_generator = RunnableSequence(prompt, model, parser)

branch_chain = RunnableBranch(
    (lambda x: len(x.split()) > 250, RunnableSequence(summarize_prompt, model, parser)),
    RunnablePassthrough()
)
```

**Conditional Logic**:

- **Condition**: `lambda x: len(x.split()) > 250`
- **If True**: Summarize the report using `summarize_prompt`
- **If False**: Pass through unchanged with `RunnablePassthrough()`

### Execution Flow

```
Input: {"topic": "The impact of AI on modern society"}
↓
report_generator: Generates report
↓
branch_chain: 
┌─── Condition: len(report.split()) > 250? ───┐
│                                             │
├─ TRUE ─────────────────┬─ FALSE ───────────┤
│                        │                   │
│ summarize_prompt       │ RunnablePassthrough│
│ → model → parser       │ (return as-is)    │
│ (create summary)       │                   │
└────────────────────────┴───────────────────┘
↓
Output: Either original report or summarized version
```

### Branching Conditions

RunnableBranch supports various condition types:

- **Lambda functions**: `lambda x: condition`
- **Custom functions**: `def my_condition(x): return boolean`
- **String matching**: Check for specific content
- **Numeric thresholds**: Length, score, or count-based conditions

### How to Run

```bash
python "5. runnable_branch.py"
```

**Expected Output**: Either a full report or a summarized version (if the report exceeds 250 words).

---

## Running the Examples

### Prerequisites Check

Before running any example, ensure Ollama is running:

```bash
ollama serve
```

### Sequential Execution

Run the examples in order to understand the progression:

```bash
# Navigate to the runnables directory
cd "e:\LangChain\6. runnables"

# Run each example
python "1. runnable_sequence.py"
python "2. runnable_parellal.py"
python "3. runnable_passthrough.py"
python "4. runnable_lambda.py"
python "5. runnable_branch.py"
```

### Troubleshooting

If you encounter issues:

1. **Model not found**: Ensure models are downloaded

   ```bash
   ollama pull gemma3:1b
   ollama pull gemma2:2b
   ollama pull llama3.1
   ```

2. **Connection error**: Check if Ollama is running

   ```bash
   ollama serve
   ```

3. **Import errors**: Install required packages

   ```bash
   pip install langchain-ollama langchain-core
   ```

---

## Key Concepts Summary

### 1. RunnableSequence

- **Purpose**: Execute operations in linear order
- **Use Case**: Multi-step processing where each step depends on the previous
- **Pattern**: A → B → C → D

### 2. RunnableParallel

- **Purpose**: Execute multiple operations simultaneously
- **Use Case**: Generate different content types from same input
- **Pattern**: A → {B, C, D} (parallel execution)

### 3. RunnablePassthrough

- **Purpose**: Forward input unchanged
- **Use Case**: Preserve original data while processing copies
- **Pattern**: Input → {Original, Processed}

### 4. RunnableLambda

- **Purpose**: Integrate custom Python functions
- **Use Case**: Custom processing, calculations, transformations
- **Pattern**: Input → Custom Function → Output

### 5. RunnableBranch

- **Purpose**: Conditional execution based on criteria
- **Use Case**: Different processing paths based on content analysis
- **Pattern**: Input → Condition Check → Path A or Path B

### Best Practices

1. **Error Handling**: Always include proper error handling in custom functions
2. **Performance**: Use parallel execution when operations are independent
3. **Modularity**: Create reusable Runnable components
4. **Testing**: Test each Runnable component independently
5. **Documentation**: Comment complex chains for maintainability

### Advanced Patterns

You can combine these Runnables to create sophisticated workflows:

```python
# Complex workflow example
complex_chain = RunnableSequence(
    initial_processor,
    RunnableParallel({
        "analysis": RunnableBranch(
            (condition1, path1),
            (condition2, path2),
            default_path
        ),
        "metadata": RunnableLambda(extract_metadata),
        "original": RunnablePassthrough()
    }),
    final_processor
)
```

This README provides a comprehensive guide to understanding and using Runnables in LangChain. Each example builds upon the previous concepts, creating a learning path from basic sequential execution to complex conditional workflows.
