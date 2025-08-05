# LangChain Chains Tutorial Collection

This folder contains comprehensive tutorials demonstrating different types of chains in LangChain, showing how to connect multiple components to create powerful AI workflows.

## What are Chains in LangChain?

Chains are sequences of operations that connect prompts, models, and output parsers to create complex AI workflows. They enable you to:

- **Compose operations** - Link multiple steps together
- **Process data flows** - Transform data between components
- **Create reusable patterns** - Build modular AI applications
- **Handle complex logic** - Implement conditional and parallel processing

## 📁 Tutorial Files Overview

| Tutorial | File | Concept | Difficulty | Key Feature |
|----------|------|---------|------------|-------------|
| 1 | `1. simple_chain.py` | Sequential Chains | Beginner | Pipe operator chaining |
| 2 | `2. parallel_chain.py` | Parallel Execution | Intermediate | RunnableParallel |
| 3 | `3. conditional_chains.py` | Conditional Logic | Advanced | RunnableBranch |

---

## Tutorial 1: Simple Chain (`1. simple_chain.py`)

### 📖 What This Tutorial Teaches

This tutorial demonstrates the fundamental concept of **sequential chaining** in LangChain, where operations are executed one after another in a linear fashion.

### 🎯 Learning Objectives

- Understand basic chain creation using the pipe operator (`|`)
- Learn how to chain multiple components sequentially
- Visualize chain structure with graph representation
- Implement a two-step AI workflow

### 🔧 Code Breakdown

```python
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama

# Model setup
chat = ChatOllama(
    model="gemma3:1b",
    temperature=0.5,
)

# First prompt: Generate detailed report
prompt = PromptTemplate(
    template="Write a report on {topic}",
    input_variables=["topic"],
)

# Second prompt: Summarize the report
next_prompt = PromptTemplate(
    template="Write a short summary of the report on {topic}",
    input_variables=["topic"],
)

# Output parser to extract text content
parser = StrOutputParser()

# Create the chain using pipe operator
chain = prompt | chat | parser | next_prompt | chat | parser

# Visualize the chain structure
chain.get_graph().print_ascii()

# Execute the chain
result = chain.invoke({"topic": "the impact of AI on society"})
print(result)
```

### 🔄 Chain Flow Explanation

1. **Input**: `{"topic": "the impact of AI on society"}`
2. **First Template**: Creates detailed report prompt
3. **First Model Call**: Generates comprehensive report
4. **Parser**: Extracts text content from model response
5. **Second Template**: Creates summary prompt using report content
6. **Second Model Call**: Generates summary
7. **Final Parser**: Extracts final summary text
8. **Output**: Summarized report

### 📊 Graph Visualization

The tutorial includes ASCII graph visualization showing:

```bash
PromptInput → PromptTemplate → ChatOllama → StrOutputParser → 
PromptTemplate → ChatOllama → StrOutputParser → Output
```

### 💡 Use Cases

- **Content Creation**: Article writing followed by summarization
- **Analysis Workflows**: Data analysis followed by reporting
- **Document Processing**: Text transformation with multiple steps
- **Quality Control**: Generation followed by review/refinement

### 🎓 Key Learning Points

- **Pipe Operator**: The `|` operator chains components seamlessly
- **Sequential Execution**: Each step depends on the previous one
- **Component Compatibility**: Output of one component becomes input of next
- **Graph Visualization**: `get_graph().print_ascii()` shows chain structure

---

## Tutorial 2: Parallel Chain (`2. parallel_chain.py`)

### 📖 What This Tutorial Teaches

This tutorial demonstrates **parallel execution** using `RunnableParallel`, where multiple operations run simultaneously and their results are combined.

### 🎯 Learning Objectives

- Master parallel chain execution with `RunnableParallel`
- Learn to use different models for different tasks
- Understand how to merge parallel results
- Optimize performance through concurrent processing

### 🔧 Code Breakdown

```python
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

# Different models for different tasks
chat1 = ChatOllama(model="gemma3:1b", temperature=0.5)
chat2 = ChatOllama(model="llama3.2:latest", temperature=0.2)

# Parallel prompt templates
prompt1 = PromptTemplate(
    template="Generate short and simple notes from the following text: \n {topic}",
    input_variables=["topic"],
)

prompt2 = PromptTemplate(
    template="Write 5 questions about the notes from the information on {topic}",
    input_variables=["topic"],
)

# Merging prompt template
prompt3 = PromptTemplate(
    template="Merge the provided notes and questions into a single document \n the notes are: {notes} \n the questions are: {questions}",
    input_variables=["notes", "questions"],
)

parser = StrOutputParser()

# Create parallel chain
parallel_chain = RunnableParallel({
    "notes": prompt1 | chat1 | parser,
    "questions": prompt2 | chat2 | parser,
})

# Create merging chain
merged_chain = prompt3 | chat1 | parser

# Combine parallel and sequential operations
final_output = parallel_chain | merged_chain

# Visualize and execute
final_output.get_graph().print_ascii()
result = final_output.invoke({"topic": sample_text})
```

### 🔄 Parallel Flow Explanation

1. **Input**: Same topic goes to both parallel branches
2. **Parallel Branch 1**: `prompt1 → chat1 → parser` (generates notes)
3. **Parallel Branch 2**: `prompt2 → chat2 → parser` (generates questions)
4. **Parallel Execution**: Both branches run simultaneously
5. **Result Merging**: `{"notes": "...", "questions": "..."}` combined
6. **Final Processing**: Merged prompt combines both outputs
7. **Output**: Single unified document

### 📊 Graph Visualization

Shows parallel execution structure:

```terminal
Input
├── Parallel Branch 1: notes generation
└── Parallel Branch 2: questions generation
    ↓
Merge Results → Final Processing → Output
```

### ⚡ Performance Benefits

- **Time Efficiency**: Parallel execution reduces total processing time
- **Resource Optimization**: Different models for specialized tasks
- **Concurrent Processing**: CPU/GPU utilization optimization
- **Scalability**: Easy to add more parallel branches

### 💡 Use Cases

- **Content Analysis**: Generate multiple perspectives simultaneously
- **Research Workflows**: Create summaries, questions, and analysis in parallel
- **Multi-format Output**: Generate different content formats concurrently
- **Comparative Analysis**: Use different models to compare outputs

### 🎓 Key Learning Points

- **RunnableParallel**: Dictionary structure defines parallel operations
- **Resource Management**: Different models for different tasks
- **Result Combination**: Parallel outputs merged into single structure
- **Performance Optimization**: Concurrent execution for independent operations

---

## Tutorial 3: Conditional Chains (`3. conditional_chains.py`)

### 📖 What This Tutorial Teaches

This tutorial demonstrates **conditional logic** using `RunnableBranch`, where the execution path depends on the content or classification of the input.

### 🎯 Learning Objectives

- Implement conditional logic with `RunnableBranch`
- Use structured output for decision making
- Create adaptive AI workflows
- Handle dynamic routing based on content analysis

### 🔧 Code Breakdown

```python
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableBranch, RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from pydantic import BaseModel, Field
from typing import Literal

# Define structured output for sentiment
class Sentiment(BaseModel):
    sentiment: Literal["positive", "negative"] = Field(
        ..., description="The sentiment of the feedback."
    )

# Setup parsers
parser = PydanticOutputParser(pydantic_object=Sentiment)
parser_out = StrOutputParser()

# Sentiment analysis prompt
prompt = PromptTemplate(
    template="Write only the classification of the sentiment of the following feedback text as either 'positive' or 'negative': \n{feedback} \n{formatted_instructions}",
    input_variables=["feedback"],
    partial_variables={"formatted_instructions": parser.get_format_instructions()},
)

chat = ChatOllama(model="llama3.1", temperature=0.5)

# Sentiment analysis chain
sentiment_chain = prompt | chat | parser

# Response prompts for different sentiments
prompt1 = PromptTemplate(
    template="Write only one feedback response to the following negative sentiment: \n{feedback}",
    input_variables=["feedback"],
)

prompt2 = PromptTemplate(
    template="Write only one feedback response to the following positive sentiment: \n{feedback}",
    input_variables=["feedback"],
)

# Conditional branching
runnable_branch = RunnableBranch(
    # (condition, runnable)
    (lambda x: x.sentiment == "negative", prompt1 | chat | parser_out),
    (lambda x: x.sentiment == "positive", prompt2 | chat | parser_out),
    RunnableLambda(lambda x: "Could not find sentiment")  # Default case
)

# Complete conditional chain
chain_with_branch = sentiment_chain | runnable_branch

# Visualize and execute
chain_with_branch.get_graph().print_ascii()
result = chain_with_branch.invoke({
    "feedback": "I love the new features in the latest update, they are fantastic!"
})
```

### 🔄 Conditional Flow Explanation

1. **Input**: Feedback text for sentiment analysis
2. **Sentiment Analysis**: Classify sentiment as positive/negative
3. **Structured Output**: Pydantic model ensures valid sentiment values
4. **Conditional Routing**: RunnableBranch routes based on sentiment
5. **Branch Selection**:
   - If negative → negative response chain
   - If positive → positive response chain
   - If unknown → default message
6. **Response Generation**: Appropriate response based on sentiment
7. **Output**: Tailored response to the detected sentiment

### 📊 Graph Visualization

Shows conditional branching structure:

```
Input → Sentiment Analysis → Branch Decision
                            ├── Positive Response
                            ├── Negative Response
                            └── Default Response
```

### 🧠 Intelligent Features

- **Dynamic Routing**: Execution path changes based on content
- **Structured Decision Making**: Pydantic models for reliable conditions
- **Fallback Handling**: Default case for unexpected inputs
- **Adaptive Responses**: Different strategies for different scenarios

### 💡 Use Cases

- **Customer Service**: Route based on sentiment/urgency
- **Content Moderation**: Different actions for different content types
- **Personalization**: Adapt responses based on user preferences
- **Quality Control**: Route for approval/rejection workflows

### 🎓 Key Learning Points

- **RunnableBranch**: Tuple format `(condition, runnable)` for branches
- **Lambda Functions**: Condition functions evaluate input data
- **Pydantic Integration**: Structured output enables reliable conditions
- **Default Handling**: Always include fallback for unknown cases

---

## 🔄 Chain Types Comparison

| Feature | Simple Chain | Parallel Chain | Conditional Chain |
|---------|-------------|----------------|-------------------|
| **Execution** | Sequential | Concurrent | Branching |
| **Performance** | Standard | High | Variable |
| **Complexity** | Low | Medium | High |
| **Use Case** | Linear workflows | Independent operations | Dynamic routing |
| **Decision Making** | None | None | Content-based |
| **Resource Usage** | Single path | Multiple resources | Selective execution |

## 🛠️ Common Components Across Tutorials

### Models Used

```python
# Different models for different purposes
ChatOllama(model="gemma3:1b", temperature=0.5)     # Fast, general
ChatOllama(model="llama3.2:latest", temperature=0.2)  # Precise
ChatOllama(model="llama3.1", temperature=0.5)      # Balanced
```

### Prompt Templates

```python
PromptTemplate(
    template="Your instruction with {variable}",
    input_variables=["variable"]
)
```

### Output Parsers

```python
StrOutputParser()  # For text output
PydanticOutputParser(pydantic_object=ModelClass)  # For structured output
```

## 🚀 Running All Tutorials

### Prerequisites

```bash
pip install langchain langchain-ollama pydantic
```

### Execute Individual Tutorials

```bash
# Tutorial 1: Simple sequential chain
python "1. simple_chain.py"

# Tutorial 2: Parallel execution
python "2. parallel_chain.py"

# Tutorial 3: Conditional logic
python "3. conditional_chains.py"
```

## 📈 Learning Progression Path

### **Beginner Level** → Tutorial 1 (Simple Chain)

- **Focus**: Understanding basic chaining concepts
- **Skills**: Pipe operator usage, sequential flow
- **Time**: 30-45 minutes

### **Intermediate Level** → Tutorial 2 (Parallel Chain)

- **Focus**: Performance optimization and concurrent execution
- **Skills**: RunnableParallel, resource management
- **Time**: 45-60 minutes

### **Advanced Level** → Tutorial 3 (Conditional Chain)

- **Focus**: Intelligent decision making and adaptive workflows
- **Skills**: RunnableBranch, structured output, dynamic routing
- **Time**: 60-90 minutes

## 💡 Best Practices from All Tutorials

### 1. **Chain Design Principles**

- Keep individual components focused and reusable
- Use appropriate parsers for expected output types
- Include error handling and fallback mechanisms
- Visualize chains before deployment

### 2. **Performance Optimization**

- Use parallel chains when operations are independent
- Choose appropriate models for each specific task
- Optimize prompts for better model performance
- Monitor execution time and resource usage

### 3. **Code Organization**

- Create modular, reusable chain components
- Use descriptive variable names and comments
- Separate configuration from logic
- Document complex conditional logic

## 🔧 Advanced Integration Patterns

### Combining All Chain Types

```python
# Complex workflow combining all patterns
def create_advanced_workflow():
    # Simple chain for preprocessing
    preprocess = prompt | model | parser
    
    # Parallel chains for analysis
    parallel_analysis = RunnableParallel({
        "sentiment": sentiment_chain,
        "summary": summary_chain,
        "questions": question_chain
    })
    
    # Conditional routing based on results
    routing = RunnableBranch(
        (lambda x: x["sentiment"].sentiment == "negative", negative_handler),
        (lambda x: x["sentiment"].sentiment == "positive", positive_handler),
        default_handler
    )
    
    # Complete workflow
    return preprocess | parallel_analysis | routing
```

## 🎯 Real-World Applications

### **Content Management System**

- **Simple**: Content creation → editing
- **Parallel**: Generate multiple content formats simultaneously
- **Conditional**: Route content based on type/audience

### **Customer Support Platform**

- **Simple**: Query processing → response generation
- **Parallel**: Sentiment + intent + entity analysis
- **Conditional**: Route to appropriate support level

### **Research Analysis Tool**

- **Simple**: Data ingestion → processing → reporting
- **Parallel**: Multiple analysis methods on same data
- **Conditional**: Analysis routing based on data type

## 🔍 Debugging and Troubleshooting

### Common Issues Across All Tutorials

1. **Chain Execution Failures**

   ```python
   # Debug with intermediate outputs
   try:
       result = chain.invoke(input_data)
   except Exception as e:
       print(f"Chain failed: {e}")
       # Add logging between components
   ```

2. **Performance Bottlenecks**

   ```python
   # Profile chain execution
   import time
   start_time = time.time()
   result = chain.invoke(input_data)
   execution_time = time.time() - start_time
   ```

3. **Unexpected Outputs**

   ```python
   # Use graph visualization
   chain.get_graph().print_ascii()
   
   # Add intermediate logging
   def debug_step(x):
       print(f"Intermediate result: {x}")
       return x
   
   debug_chain = component1 | RunnableLambda(debug_step) | component2
   ```

## 📚 Key Takeaways from All Tutorials

1. **Sequential Chains** provide the foundation for all LangChain operations
2. **Parallel Execution** significantly improves performance for independent operations
3. **Conditional Logic** enables intelligent, adaptive AI systems
4. **Graph Visualization** is essential for understanding and debugging chains
5. **Component Composition** allows building complex workflows from simple parts
6. **Error Handling** and fallback mechanisms are crucial for production systems

## 🔗 Related Advanced Topics

- **Custom Runnables**: Building your own chain components
- **Memory Integration**: Adding conversation memory to chains
- **Tool Integration**: Incorporating external tools into chains
- **Async Execution**: Non-blocking chain execution
- **Chain Persistence**: Saving and loading chain configurations

This comprehensive tutorial collection provides everything needed to master LangChain chains, from basic concepts to advanced patterns used in production AI applications.

**Parallel execution with RunnableParallel for concurrent processing**

#### What it demonstrates

- Running multiple chains simultaneously using `RunnableParallel`
- Merging results from parallel operations
- Using different models for different tasks
- Complex data flow management

#### Key concepts

```python
# Parallel chain structure
parallel_chain = RunnableParallel({
    "notes": prompt1 | chat1 | parser,
    "questions": prompt2 | chat2 | parser,
})

# Merge parallel results
final_chain = parallel_chain | merged_chain
```

#### Benefits

- **Performance improvement** - Parallel execution reduces total time
- **Resource optimization** - Different models for different tasks
- **Complex workflows** - Generate multiple outputs simultaneously

#### Use case

- **Content analysis** - Generate notes and questions simultaneously
- **Multi-perspective processing** - Different models analyzing the same input
- **Efficiency optimization** - When operations can run independently

---

### 3. **Conditional Chains** (`3. conditional_chains.py`)

**Dynamic routing with RunnableBranch for conditional logic**

#### What it demonstrates

- Implementing conditional logic with `RunnableBranch`
- Dynamic routing based on model outputs
- Using structured output for decision making
- Creating responsive AI workflows

#### Key concepts

```python
# Sentiment analysis with structured output
class Sentiment(BaseModel):
    sentiment: Literal["positive", "negative"]

# Conditional branching
runnable_branch = RunnableBranch(
    (lambda x: x.sentiment == "negative", negative_response_chain),
    (lambda x: x.sentiment == "positive", positive_response_chain),
    default_chain  # Fallback option
)

# Complete conditional chain
conditional_chain = sentiment_chain | runnable_branch
```

#### Use case

- **Dynamic responses** - Different actions based on input classification
- **Content routing** - Send content to appropriate processing pipeline
- **Intelligent workflows** - Adaptive behavior based on context

---

## 🔄 Chain Types Comparison

| Chain Type | Execution | Use Case | Complexity | Performance |
|------------|-----------|----------|------------|-------------|
| **Simple** | Sequential | Linear workflows | Low | Standard |
| **Parallel** | Concurrent | Independent operations | Medium | High |
| **Conditional** | Branching | Dynamic routing | High | Variable |

## 🛠️ Core Components

### 1. **Prompt Templates**

```python
prompt = PromptTemplate(
    template="Write a report on {topic}",
    input_variables=["topic"]
)
```

### 2. **Models**

```python
chat = ChatOllama(
    model="gemma3:1b",
    temperature=0.5
)
```

### 3. **Output Parsers**

```python
# String parser for text output
str_parser = StrOutputParser()

# Structured parser for validated output
pydantic_parser = PydanticOutputParser(pydantic_object=MyModel)
```

### 4. **Chain Operators**

```python
# Pipe operator for sequential chaining
chain = component1 | component2 | component3

# Parallel execution
parallel = RunnableParallel({"key1": chain1, "key2": chain2})

# Conditional branching
branch = RunnableBranch((condition, chain), default_chain)
```

## 📊 Graph Visualization

Each tutorial includes graph visualization using:

```python
chain.get_graph().print_ascii()
```

This shows the execution flow and helps understand:

- **Component connections** - How data flows between components
- **Execution order** - Sequential vs parallel operations
- **Decision points** - Where branching occurs
- **Data transformations** - Where outputs become inputs

## 🎯 Learning Progression

### **Beginner** → Simple Chain

- Understand basic chaining concepts
- Learn pipe operator usage
- See sequential data flow

### **Intermediate** → Parallel Chain

- Master concurrent execution
- Handle multiple data streams
- Optimize performance

### **Advanced** → Conditional Chain

- Implement dynamic logic
- Use structured outputs for decisions
- Create adaptive workflows

## 🚀 Running the Examples

### Prerequisites

```bash
pip install langchain langchain-ollama pydantic
```

### Execute Examples

```bash
# Simple chain
python "1. simple_chain.py"

# Parallel chain
python "2. parallel_chain.py"

# Conditional chain
python "3. conditional_chains.py"
```

## 💡 Best Practices

### 1. **Chain Design**

- **Keep chains focused** - Each chain should have a clear purpose
- **Use appropriate parsers** - Match parser to expected output
- **Handle errors gracefully** - Add error handling for robust chains

### 2. **Performance Optimization**

- **Use parallel chains** when operations are independent
- **Choose appropriate models** for each task
- **Optimize prompts** for better model performance

### 3. **Code Organization**

- **Modular design** - Create reusable chain components
- **Clear naming** - Use descriptive variable names
- **Documentation** - Comment complex chain logic

## 🔧 Advanced Patterns

### Chain Composition

```python
# Build complex chains from simpler ones
analysis_chain = prompt | model | parser
summary_chain = summary_prompt | model | parser
complete_workflow = analysis_chain | summary_chain
```

### Error Handling

```python
try:
    result = chain.invoke(input_data)
except Exception as e:
    print(f"Chain execution failed: {e}")
    # Implement fallback logic
```

### Dynamic Chain Building

```python
def build_chain(use_parallel=False):
    if use_parallel:
        return parallel_chain
    else:
        return simple_chain
```

## 🎨 Use Cases by Industry

### **Content Creation**

- **Simple**: Article writing → editing
- **Parallel**: Generate multiple content variants
- **Conditional**: Content routing by audience type

### **Data Analysis**

- **Simple**: Data processing → reporting
- **Parallel**: Multiple analysis methods
- **Conditional**: Analysis routing by data type

### **Customer Service**

- **Simple**: Query understanding → response generation
- **Parallel**: Sentiment + intent analysis
- **Conditional**: Response routing by sentiment

## 🔍 Troubleshooting

### Common Issues

1. **Chain execution failures**
   - Check input/output compatibility between components
   - Verify model availability and configuration

2. **Performance bottlenecks**
   - Consider parallel execution for independent operations
   - Optimize prompts and model parameters

3. **Unexpected outputs**
   - Use graph visualization to understand data flow
   - Add intermediate logging for debugging

## 📚 Key Takeaways

1. **Chains enable complex AI workflows** by connecting simple components
2. **Different chain types** serve different architectural needs
3. **Parallel execution** improves performance for independent operations
4. **Conditional logic** creates adaptive, intelligent systems
5. **Graph visualization** helps understand and debug chain behavior

## 🔗 Related Concepts

- **Prompt Engineering** - Crafting effective prompts for chains
- **Output Parsing** - Structuring model outputs for chain compatibility
- **Model Selection** - Choosing appropriate models for each chain step
- **Performance Optimization** - Techniques for efficient chain execution

This tutorial series provides a comprehensive foundation for building sophisticated AI applications using LangChain's powerful chaining capabilities.
