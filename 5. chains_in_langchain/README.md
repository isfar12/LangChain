# LangChain Chains - Complete Step-by-Step Tutorial

This comprehensive guide breaks down LangChain chains into easy-to-understand sections, progressing from basic concepts to advanced implementations. Each section builds upon the previous one, making complex topics accessible to beginners.

## 🎯 What You'll Learn

By the end of this tutorial, you'll master:
- **Basic chaining concepts** and the pipe operator
- **Sequential processing** with simple chains
- **Parallel execution** for performance optimization
- **Conditional logic** for intelligent decision-making
- **Graph visualization** for debugging and understanding
- **Real-world applications** and best practices

---

## 🧠 **PART 1: Understanding Chains - The Foundation**

### **What are Chains?**

Think of chains as **assembly lines for AI processing**. Just like a factory assembly line, each step:
- ✅ **Receives input** from the previous step
- ✅ **Processes the data** in a specific way
- ✅ **Passes output** to the next step
- ✅ **Works together** to create a final result

### **Why Use Chains?**

Instead of writing complex, monolithic functions, chains let you:

**🔄 Break complex tasks into simple steps**
```python
# Instead of: huge_complex_function(input) → output
# Use: step1 | step2 | step3 | step4 → output
```

**🔧 Create reusable components**
```python
# Define once, use many times
email_parser = PromptTemplate(...) | model | StrOutputParser()
# Use in different chains: customer_service_chain, marketing_chain, etc.
```

**🎯 Make debugging easier**
```python
# You can test each step independently
step1_result = step1.invoke(input)  # Test just this step
step2_result = step2.invoke(step1_result)  # Test the next step
```

### **Core Building Blocks**

Every chain uses these fundamental components:

#### **1. Prompt Templates** - The Instructions
```python
from langchain_core.prompts import PromptTemplate

# Template with placeholders
prompt = PromptTemplate(
    template="Write a {style} summary of: {text}",
    input_variables=["style", "text"]
)

# When invoked: "Write a brief summary of: [your text here]"
```

#### **2. Models** - The AI Brain
```python
from langchain_ollama import ChatOllama

# The AI model that processes your prompts
model = ChatOllama(
    model="gemma3:1b",        # Which AI model to use
    temperature=0.5           # Creativity level (0=focused, 1=creative)
)
```

#### **3. Output Parsers** - The Formatter
```python
from langchain_core.output_parsers import StrOutputParser

# Converts model output to usable format
parser = StrOutputParser()  # Extracts plain text from model response
```

#### **4. The Pipe Operator** - The Connector
```python
# The "|" symbol connects components
chain = prompt | model | parser
#      ↑       ↑       ↑
#   step 1  step 2  step 3
```

### **How Data Flows Through Chains**

```python
# Input data flows left to right
input_data = {"style": "brief", "text": "Long article about AI..."}
     ↓
prompt.invoke(input_data)  # Creates formatted prompt
     ↓
model.invoke(formatted_prompt)  # AI processes the prompt
     ↓
parser.invoke(model_response)  # Extracts clean text
     ↓
final_result  # Clean, formatted output
```

---

## 🚀 **PART 2: Simple Chains - Your First Chain**

Let's build your first chain step by step using `1. simple_chain.py`.

### **Step 1: Import Required Components**

```python
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
```

**What each import does:**
- `StrOutputParser`: Converts AI response to clean text
- `PromptTemplate`: Creates reusable prompt templates
- `ChatOllama`: Connects to local Ollama AI models

### **Step 2: Set Up the AI Model**

```python
chat = ChatOllama(
    model="gemma3:1b",     # Fast, lightweight model
    temperature=0.5,       # Balanced creativity
)
```

**Model selection explained:**
- `gemma3:1b`: 1 billion parameter model (fast, good for basic tasks)
- `temperature=0.5`: Mid-range creativity (0=very focused, 1=very creative)

### **Step 3: Create Prompt Templates**

```python
# First prompt: Generate detailed content
prompt = PromptTemplate(
    template="Write a report on {topic}",
    input_variables=["topic"],
)

# Second prompt: Summarize the content
next_prompt = PromptTemplate(
    template="Write a short summary of the report on {topic}",
    input_variables=["topic"],
)
```

**Template breakdown:**
- `{topic}`: Placeholder that gets replaced with actual values
- `input_variables=["topic"]`: Tells LangChain which placeholders to expect

### **Step 4: Set Up Output Parser**

```python
parser = StrOutputParser()
```

**What this does:**
- Takes the AI model's complex response object
- Extracts just the text content
- Returns a clean string you can use

### **Step 5: Build the Chain**

```python
chain = prompt | chat | parser | next_prompt | chat | parser
```

**Chain flow visualization:**
```
Input → Prompt1 → Model → Parser → Prompt2 → Model → Parser → Output
  ↓        ↓        ↓       ↓        ↓        ↓       ↓        ↓
topic → "Write... → AI → Clean → "Summ... → AI → Clean → Final
        report"     text    text     arize"    text   text   result
```

### **Step 6: Visualize the Chain**

```python
chain.get_graph().print_ascii()
```

**ASCII output shows:**
```
     +-------------+       
     | PromptInput |       
     +-------------+       
            ↓
    +----------------+     
    | PromptTemplate |     
    +----------------+     
            ↓
      +------------+       
      | ChatOllama |       
      +------------+       
            ↓
   +-----------------+
   | StrOutputParser |
   +-----------------+
            ↓
    +----------------+
    | PromptTemplate |
    +----------------+
            ↓
      +------------+
      | ChatOllama |
      +------------+
            ↓
   +-----------------+
   | StrOutputParser |
   +-----------------+
```

### **Step 7: Execute the Chain**

```python
result = chain.invoke({"topic": "the impact of AI on society"})
print(result)
```

**What happens:**
1. `{"topic": "the impact of AI on society"}` enters the chain
2. First prompt becomes: "Write a report on the impact of AI on society"
3. AI generates a detailed report
4. Parser extracts clean text
5. Second prompt becomes: "Write a short summary of the report on the impact of AI on society"
6. AI generates a summary
7. Parser extracts final clean text
8. You get a summarized report

### **🎓 Key Learning Points**

✅ **Pipe operator (`|`)** connects components sequentially
✅ **Each component** transforms data and passes it to the next
✅ **Templates** make prompts reusable and dynamic
✅ **Parsers** clean up AI outputs for your application
✅ **Graph visualization** helps understand and debug chains

### **🛠️ Try It Yourself**

Modify the example to create different workflows:

```python
# Creative writing chain
creative_prompt = PromptTemplate(
    template="Write a creative story about {topic}",
    input_variables=["topic"]
)
critique_prompt = PromptTemplate(
    template="Provide constructive feedback on this story: {topic}",
    input_variables=["topic"]
)
creative_chain = creative_prompt | chat | parser | critique_prompt | chat | parser

# Business analysis chain
analysis_prompt = PromptTemplate(
    template="Analyze the business impact of {topic}",
    input_variables=["topic"]
)
action_prompt = PromptTemplate(
    template="Suggest 3 action items based on this analysis: {topic}",
    input_variables=["topic"]
)
business_chain = analysis_prompt | chat | parser | action_prompt | chat | parser
```

---

## ⚡ **PART 3: Parallel Chains - Speed and Efficiency**

Now let's explore parallel processing using `2. parallel_chain.py` to dramatically improve performance.

### **Why Parallel Chains?**

**Sequential processing (what we did before):**
```
Step 1 (5 seconds) → Step 2 (5 seconds) → Step 3 (5 seconds) = 15 seconds total
```

**Parallel processing (what we'll do now):**
```
Step 1 (5 seconds) \
                    → Step 3 (5 seconds) = 10 seconds total
Step 2 (5 seconds) /
```

**Performance gain: 33% faster!**

### **Step 1: Import Parallel Components**

```python
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel  # ← New import
```

**New component:**
- `RunnableParallel`: Runs multiple chains simultaneously

### **Step 2: Set Up Multiple Models**

```python
# Model 1: Fast and efficient for note-taking
chat1 = ChatOllama(
    model="gemma3:1b",         # Lightweight model
    temperature=0.5,           # Balanced creativity
)

# Model 2: More precise for question generation
chat2 = ChatOllama(
    model="llama3.2:latest",   # More sophisticated model
    temperature=0.2,           # Lower temperature for precision
)
```

**Strategy explanation:**
- **Different models for different tasks** - Use the best tool for each job
- **Specialized configurations** - Different temperatures for different goals

### **Step 3: Create Specialized Prompts**

```python
# Prompt 1: Generate notes (for chat1)
prompt1 = PromptTemplate(
    template="Generate short and simple notes from the following text: \n {topic}",
    input_variables=["topic"],
)

# Prompt 2: Generate questions (for chat2)
prompt2 = PromptTemplate(
    template="Write 5 questions about the notes from the information on {topic}",
    input_variables=["topic"],
)

# Prompt 3: Merge results (combines parallel outputs)
prompt3 = PromptTemplate(
    template="Merge the provided notes and questions into a single document \n the notes are: {notes} \n the questions are: {questions}",
    input_variables=["notes", "questions"],  # ← Notice: two inputs!
)
```

**Key insight:**
- `prompt3` takes **two inputs** (`notes` and `questions`)
- This is how we combine the results from parallel processing

### **Step 4: Build Individual Chains**

```python
parser = StrOutputParser()

# Chain 1: Text → Notes
notes_chain = prompt1 | chat1 | parser

# Chain 2: Text → Questions  
questions_chain = prompt2 | chat2 | parser
```

**What we have so far:**
- Two independent chains that can run simultaneously
- Each produces different output from the same input

### **Step 5: Create Parallel Chain**

```python
parallel_chain = RunnableParallel({
    "notes": prompt1 | chat1 | parser,
    "questions": prompt2 | chat2 | parser,
})
```

**RunnableParallel structure:**
```python
{
    "output_key_1": chain_1,
    "output_key_2": chain_2,
    # Add as many parallel chains as needed
}
```

**What happens when executed:**
```python
# Input: {"topic": "AI in healthcare"}
# 
# Parallel execution:
# Branch 1: "notes" key gets → notes about AI in healthcare
# Branch 2: "questions" key gets → questions about AI in healthcare
#
# Output: {
#     "notes": "AI transforms healthcare by...",
#     "questions": "1. How does AI improve diagnosis? 2. What are the risks?..."
# }
```

### **Step 6: Create Merging Chain**

```python
merged_chain = prompt3 | chat1 | parser
```

**This chain:**
- Takes the dictionary output from `parallel_chain`
- Uses `prompt3` to combine notes and questions
- Produces a single unified document

### **Step 7: Connect Everything**

```python
final_output = parallel_chain | merged_chain
```

**Complete flow:**
```
Input Topic
    ↓
Parallel Processing
├── Notes Chain    (runs simultaneously)
└── Questions Chain (runs simultaneously)
    ↓
{"notes": "...", "questions": "..."}
    ↓
Merging Chain
    ↓
Final Unified Document
```

### **Step 8: Visualize Parallel Structure**

```python
final_output.get_graph().print_ascii()
```

**ASCII visualization shows:**
```
         +--------------------------------+
         | Parallel<notes,questions>Input |
         +--------------------------------+
                 **               **
              ***                   ***
            **                         **
+----------------+                +----------------+ 
| PromptTemplate |                | PromptTemplate | 
+----------------+                +----------------+ 
          *                               *
  +------------+                    +------------+   
  | ChatOllama |                    | ChatOllama |   
  +------------+                    +------------+   
          *                               *
+-----------------+              +-----------------+
| StrOutputParser |              | StrOutputParser |
+-----------------+              +-----------------+
                 **               **
                   ***         ***
                      **     **
        +---------------------------------+
        | Parallel<notes,questions>Output |
        +---------------------------------+
                          *
                 +----------------+
                 | PromptTemplate |
                 +----------------+
                          *
                   +------------+
                   | ChatOllama |
                   +------------+
                          *
                +-----------------+
                | StrOutputParser |
                +-----------------+
```

### **Step 9: Execute with Real Data**

```python
topic = '''
The Kaggle community has a lot of diversity, with members from over 100 countries 
and skill levels ranging from those learning Python through to the researchers who 
created deep neural networks. We have had competition winners with backgrounds 
ranging from computer science to English literature. However, all our users share 
a common thread: you love working with data.

As our community grows, we want to make sure that Kaggle continues to be welcoming. 
To that end, we are introducing guidelines to ensure everyone has the same 
expectations about discourse in the forums...
'''

result = final_output.invoke({"topic": topic})
print(result)
```

**What you get:**
- **Notes**: Structured summary of the Kaggle community information
- **Questions**: Thoughtful questions about the content
- **Merged document**: Professional combination of both

### **🎓 Key Learning Points**

✅ **RunnableParallel** enables simultaneous execution
✅ **Dictionary structure** organizes parallel outputs
✅ **Different models** can be optimized for different tasks
✅ **Merging chains** combine parallel results
✅ **Performance gains** come from concurrent processing

### **⚡ Performance Comparison**

```python
import time

# Sequential timing
start = time.time()
sequential_result = (prompt1 | chat1 | parser).invoke({"topic": topic})
sequential_result2 = (prompt2 | chat2 | parser).invoke({"topic": topic})
sequential_time = time.time() - start

# Parallel timing
start = time.time()
parallel_result = parallel_chain.invoke({"topic": topic})
parallel_time = time.time() - start

print(f"Sequential: {sequential_time:.2f}s")
print(f"Parallel: {parallel_time:.2f}s")
print(f"Speedup: {sequential_time/parallel_time:.2f}x")
```

### **🛠️ Advanced Parallel Patterns**

```python
# Multi-model analysis
analysis_parallel = RunnableParallel({
    "sentiment": sentiment_prompt | sentiment_model | parser,
    "summary": summary_prompt | summary_model | parser,
    "keywords": keyword_prompt | keyword_model | parser,
    "translation": translate_prompt | translate_model | parser,
})

# Multi-format output
format_parallel = RunnableParallel({
    "json": data_prompt | model | json_parser,
    "markdown": data_prompt | model | markdown_parser,
    "html": data_prompt | model | html_parser,
})

# Multi-perspective analysis
perspective_parallel = RunnableParallel({
    "technical": tech_prompt | tech_model | parser,
    "business": biz_prompt | biz_model | parser,
    "user": user_prompt | user_model | parser,
})
```

---

## 🧠 **PART 4: Conditional Chains - Smart Decision Making**

Now let's explore intelligent routing using `3. conditional_chains.py` to create AI systems that adapt their behavior.

### **Why Conditional Chains?**

Imagine a customer service system that:
- **Positive feedback** → Thank you response + feedback collection
- **Negative feedback** → Apology response + issue escalation
- **Neutral feedback** → Information request + follow-up

**One input, different paths based on content analysis!**

### **Step 1: Import Conditional Components**

```python
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableBranch, RunnableLambda  # ← New imports
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from pydantic import BaseModel, Field
from typing import Literal
```

**New components:**
- `RunnableBranch`: Routes execution based on conditions
- `RunnableLambda`: Wraps functions as chain components
- `PydanticOutputParser`: Ensures structured, validated output

### **Step 2: Define Structured Output**

```python
class Sentiment(BaseModel):
    sentiment: Literal["positive", "negative"] = Field(
        ..., 
        description="The sentiment of the feedback."
    )
```

**Why structured output?**
- **Reliability**: Only "positive" or "negative" values allowed
- **Type safety**: No parsing errors or unexpected values
- **Clear conditions**: Easy to write if/else logic

### **Step 3: Set Up Parsers**

```python
# For structured output (sentiment classification)
parser = PydanticOutputParser(pydantic_object=Sentiment)

# For text output (final responses)
parser_out = StrOutputParser()
```

**Two different parsers:**
- `PydanticOutputParser`: Validates and structures data
- `StrOutputParser`: Extracts clean text for final output

### **Step 4: Create Sentiment Analysis Chain**

```python
# Sentiment classification prompt
prompt = PromptTemplate(
    template="Write only the classification of the sentiment of the following feedback text as either 'positive' or 'negative': \n{feedback} \n{formatted_instructions}",
    input_variables=["feedback"],
    partial_variables={"formatted_instructions": parser.get_format_instructions()},
)

chat = ChatOllama(
    model="llama3.1",      # More capable model for classification
    temperature=0.5,
)

# Sentiment analysis chain
sentiment_chain = prompt | chat | parser
```

**Key components:**
- `{formatted_instructions}`: Automatically adds format instructions from Pydantic
- `partial_variables`: Pre-fills template variables
- Results in reliable sentiment classification

### **Step 5: Create Response Prompts**

```python
# Negative sentiment response
prompt1 = PromptTemplate(
    template="Write only one feedback response to the following negative sentiment: \n{feedback}",
    input_variables=["feedback"],
)

# Positive sentiment response
prompt2 = PromptTemplate(
    template="Write only one feedback response to the following positive sentiment: \n{feedback}",
    input_variables=["feedback"],
)
```

**Different responses for different sentiments:**
- **Negative**: Apologetic, solution-focused
- **Positive**: Appreciative, encouraging

### **Step 6: Build Conditional Logic**

```python
runnable_branch = RunnableBranch(
    # Format: (condition_function, chain_to_execute)
    (lambda x: x.sentiment == "negative", prompt1 | chat | parser_out),
    (lambda x: x.sentiment == "positive", prompt2 | chat | parser_out),
    RunnableLambda(lambda x: "Could not find sentiment")  # Default case
)
```

**RunnableBranch structure:**
```python
RunnableBranch(
    (condition_1, chain_1),  # If condition_1 is True, run chain_1
    (condition_2, chain_2),  # Else if condition_2 is True, run chain_2
    default_chain            # Else run default_chain
)
```

**Lambda functions explained:**
```python
# This function receives the sentiment analysis result
lambda x: x.sentiment == "negative"
#     ↑              ↑
#  input object   condition check

# x looks like: Sentiment(sentiment="negative")
# x.sentiment returns: "negative"
# Comparison returns: True or False
```

### **Step 7: Connect Sentiment Analysis to Branching**

```python
chain_with_branch = sentiment_chain | runnable_branch
```

**Complete flow:**
```
Feedback Text
     ↓
Sentiment Analysis → Sentiment(sentiment="positive")
     ↓
Conditional Branch
├── If positive → Positive response chain
├── If negative → Negative response chain
└── Else → Default message
     ↓
Final Response
```

### **Step 8: Visualize Conditional Structure**

```python
chain_with_branch.get_graph().print_ascii()
```

**ASCII shows branching:**
```
    +-------------+      
    | PromptInput |      
    +-------------+      
            *
   +----------------+    
   | PromptTemplate |    
   +----------------+    
            *
     +------------+      
     | ChatOllama |      
     +------------+      
            *
+----------------------+
| PydanticOutputParser |
+----------------------+
            *
       +--------+
       | Branch |  ← Decision point
       +--------+
            *
    +--------------+
    | BranchOutput |
    +--------------+
```

### **Step 9: Test with Different Inputs**

```python
# Positive feedback test
positive_result = chain_with_branch.invoke({
    "feedback": "I love the new features in the latest update, they are fantastic!"
})
print("Positive:", positive_result)

# Negative feedback test
negative_result = chain_with_branch.invoke({
    "feedback": "This service is terrible and the support is unhelpful!"
})
print("Negative:", negative_result)

# Edge case test
neutral_result = chain_with_branch.invoke({
    "feedback": "The weather is nice today."
})
print("Neutral:", neutral_result)
```

### **🎓 Key Learning Points**

✅ **Structured output** enables reliable decision making
✅ **Lambda functions** provide flexible condition logic
✅ **Multiple branches** handle different scenarios
✅ **Default cases** prevent system failures
✅ **Dynamic routing** creates adaptive AI behavior

### **🛠️ Advanced Conditional Patterns**

#### **Multi-level Conditions**
```python
# Complex routing based on multiple factors
advanced_branch = RunnableBranch(
    # Priority routing
    (lambda x: x.urgency == "high" and x.sentiment == "negative", priority_chain),
    (lambda x: x.sentiment == "negative", standard_negative_chain),
    (lambda x: x.sentiment == "positive" and x.score > 8, premium_positive_chain),
    (lambda x: x.sentiment == "positive", standard_positive_chain),
    default_chain
)
```

#### **Content-type Routing**
```python
# Route based on content analysis
content_router = RunnableBranch(
    (lambda x: x.content_type == "question", qa_chain),
    (lambda x: x.content_type == "complaint", complaint_chain),
    (lambda x: x.content_type == "compliment", thanks_chain),
    (lambda x: x.content_type == "request", request_chain),
    general_chain
)
```

#### **User-based Routing**
```python
# Route based on user characteristics
user_router = RunnableBranch(
    (lambda x: x.user_tier == "premium", premium_chain),
    (lambda x: x.user_tier == "basic", basic_chain),
    (lambda x: x.user_status == "new", onboarding_chain),
    standard_chain
)
```

---

## 🔧 **PART 5: Advanced Chain Patterns and Best Practices**

### **Combining All Three Types**

Here's how to create sophisticated workflows using all chain types together:

```python
def create_comprehensive_workflow():
    # 1. Simple chain for preprocessing
    preprocess_chain = cleanup_prompt | model | parser
    
    # 2. Parallel analysis
    analysis_parallel = RunnableParallel({
        "sentiment": sentiment_analysis_chain,
        "intent": intent_analysis_chain,
        "entities": entity_extraction_chain,
        "priority": priority_assessment_chain
    })
    
    # 3. Conditional routing based on analysis
    routing_branch = RunnableBranch(
        (lambda x: x["priority"]["level"] == "critical", critical_response_chain),
        (lambda x: x["sentiment"]["sentiment"] == "negative", negative_response_chain),
        (lambda x: x["intent"]["intent"] == "question", qa_response_chain),
        standard_response_chain
    )
    
    # 4. Complete workflow
    return preprocess_chain | analysis_parallel | routing_branch
```

### **Error Handling and Robustness**

```python
from langchain_core.runnables import RunnableLambda

def safe_chain_execution(chain, input_data, fallback_response="Sorry, I couldn't process that."):
    """Execute chain with error handling"""
    try:
        return chain.invoke(input_data)
    except Exception as e:
        print(f"Chain execution failed: {e}")
        return fallback_response

# Wrap chains with error handling
safe_chain = RunnableLambda(lambda x: safe_chain_execution(your_chain, x))
```

### **Performance Monitoring**

```python
import time
from contextlib import contextmanager

@contextmanager
def timer(description="Operation"):
    start = time.time()
    yield
    end = time.time()
    print(f"{description} took {end - start:.2f} seconds")

# Usage
with timer("Chain execution"):
    result = chain.invoke(input_data)
```

### **Chain Testing and Validation**

```python
def test_chain(chain, test_cases):
    """Test chain with multiple inputs"""
    results = []
    for i, test_case in enumerate(test_cases):
        try:
            result = chain.invoke(test_case["input"])
            results.append({
                "test": i + 1,
                "input": test_case["input"],
                "output": result,
                "expected": test_case.get("expected"),
                "passed": result == test_case.get("expected") if "expected" in test_case else "N/A"
            })
        except Exception as e:
            results.append({
                "test": i + 1,
                "input": test_case["input"],
                "error": str(e),
                "passed": False
            })
    return results

# Test cases
test_cases = [
    {"input": {"feedback": "Great service!"}, "expected": "positive"},
    {"input": {"feedback": "Terrible experience!"}, "expected": "negative"},
]

test_results = test_chain(sentiment_chain, test_cases)
```

---

## 📊 **PART 6: Performance Optimization and Production Tips**

### **Model Selection Strategy**

| Task Type | Recommended Model | Temperature | Why |
|-----------|------------------|-------------|-----|
| **Classification** | llama3.1 | 0.1-0.3 | Accuracy over creativity |
| **Creative Writing** | llama3.2:latest | 0.7-0.9 | Creativity important |
| **Summarization** | gemma3:1b | 0.3-0.5 | Fast and adequate |
| **Question Answering** | llama3.1 | 0.2-0.4 | Factual accuracy |
| **Sentiment Analysis** | gemma3:1b | 0.1-0.2 | Consistency crucial |

### **Chain Optimization Techniques**

#### **1. Use Appropriate Parallelization**
```python
# Good: Independent operations
parallel_chain = RunnableParallel({
    "summary": summary_chain,
    "sentiment": sentiment_chain,  # These don't depend on each other
    "keywords": keyword_chain
})

# Bad: Dependent operations (use sequential instead)
# Don't parallelize: analysis → response_based_on_analysis
```

#### **2. Optimize Prompt Templates**
```python
# Good: Specific and clear
good_prompt = PromptTemplate(
    template="Extract exactly 3 key points from this text as a bullet list: {text}",
    input_variables=["text"]
)

# Bad: Vague and unclear
bad_prompt = PromptTemplate(
    template="Tell me about {text}",  # Too vague
    input_variables=["text"]
)
```

#### **3. Use Caching for Repeated Calls**
```python
from functools import lru_cache

@lru_cache(maxsize=100)
def cached_chain_invoke(input_text):
    return chain.invoke({"text": input_text})
```

### **Memory Management**

```python
# For long-running applications
import gc

def process_batch(inputs):
    results = []
    for i, input_data in enumerate(inputs):
        result = chain.invoke(input_data)
        results.append(result)
        
        # Periodic cleanup
        if i % 100 == 0:
            gc.collect()  # Force garbage collection
            
    return results
```

---

## 🎯 **PART 7: Real-World Applications and Use Cases**

### **Customer Service Automation**

```python
def build_customer_service_chain():
    # Analyze incoming message
    analysis_parallel = RunnableParallel({
        "sentiment": sentiment_analysis_chain,
        "intent": intent_classification_chain,
        "urgency": urgency_assessment_chain,
        "language": language_detection_chain
    })
    
    # Route based on analysis
    routing_branch = RunnableBranch(
        # Critical issues go to human agents
        (lambda x: x["urgency"]["level"] == "critical", human_escalation_chain),
        
        # Negative sentiment gets priority treatment
        (lambda x: x["sentiment"]["sentiment"] == "negative", 
         priority_negative_response_chain),
        
        # FAQ questions get automated responses
        (lambda x: x["intent"]["intent"] == "faq", automated_faq_chain),
        
        # Everything else gets standard response
        standard_response_chain
    )
    
    return analysis_parallel | routing_branch

# Usage
customer_service = build_customer_service_chain()
response = customer_service.invoke({
    "message": "I'm furious! My order hasn't arrived and it's been a week!"
})
```

### **Content Creation Pipeline**

```python
def build_content_pipeline():
    # Generate multiple content variations in parallel
    content_parallel = RunnableParallel({
        "blog_post": blog_writing_chain,
        "social_media": social_media_chain,
        "newsletter": newsletter_chain,
        "summary": summary_chain
    })
    
    # Quality check and routing
    quality_branch = RunnableBranch(
        (lambda x: x["quality_score"] < 7, regeneration_chain),
        approval_chain
    )
    
    return content_parallel | quality_assessment_chain | quality_branch

# Usage
content_pipeline = build_content_pipeline()
content_package = content_pipeline.invoke({
    "topic": "AI in Healthcare",
    "target_audience": "general public",
    "tone": "informative but accessible"
})
```

### **Data Analysis Workflow**

```python
def build_analysis_workflow():
    # Parallel analysis approaches
    analysis_parallel = RunnableParallel({
        "statistical": statistical_analysis_chain,
        "qualitative": qualitative_analysis_chain,
        "trend": trend_analysis_chain,
        "anomaly": anomaly_detection_chain
    })
    
    # Synthesis based on findings
    synthesis_branch = RunnableBranch(
        (lambda x: x["anomaly"]["found"], anomaly_report_chain),
        (lambda x: x["trend"]["significant"], trend_report_chain),
        standard_report_chain
    )
    
    return analysis_parallel | synthesis_branch

# Usage
data_analyzer = build_analysis_workflow()
analysis_report = data_analyzer.invoke({
    "data": sales_data,
    "time_period": "Q4 2024",
    "focus_areas": ["growth", "customer_satisfaction", "product_performance"]
})
```

---

## 🧪 **PART 8: Testing and Debugging Chains**

### **Debugging Chain Execution**

```python
def debug_chain(chain, input_data):
    """Step-by-step chain execution with debugging"""
    print(f"🔍 Debugging chain with input: {input_data}")
    print(f"📊 Chain structure:")
    chain.get_graph().print_ascii()
    
    try:
        result = chain.invoke(input_data)
        print(f"✅ Chain executed successfully")
        print(f"📤 Final output: {result}")
        return result
    except Exception as e:
        print(f"❌ Chain execution failed: {e}")
        print(f"🔧 Debugging tips:")
        print("  1. Check input format matches template variables")
        print("  2. Verify model availability")
        print("  3. Test individual chain components")
        raise

# Usage
debug_chain(your_chain, test_input)
```

### **Component-by-Component Testing**

```python
def test_chain_components():
    # Test each component individually
    print("Testing prompt template...")
    formatted_prompt = prompt.invoke({"topic": "AI"})
    print(f"Formatted prompt: {formatted_prompt}")
    
    print("Testing model...")
    model_response = model.invoke(formatted_prompt)
    print(f"Model response: {model_response}")
    
    print("Testing parser...")
    parsed_output = parser.invoke(model_response)
    print(f"Parsed output: {parsed_output}")
    
    print("Testing full chain...")
    full_result = chain.invoke({"topic": "AI"})
    print(f"Full chain result: {full_result}")

test_chain_components()
```

### **Performance Profiling**

```python
import cProfile
import pstats

def profile_chain(chain, input_data):
    """Profile chain execution performance"""
    profiler = cProfile.Profile()
    profiler.enable()
    
    result = chain.invoke(input_data)
    
    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(10)  # Top 10 slowest functions
    
    return result

# Usage
result = profile_chain(your_chain, test_input)
```

---

## 🚀 **PART 9: Next Steps and Advanced Topics**

### **What You've Mastered**

After completing this tutorial, you now understand:
- ✅ **Sequential chains** for step-by-step processing
- ✅ **Parallel chains** for performance optimization  
- ✅ **Conditional chains** for intelligent decision making
- ✅ **Component integration** and data flow management
- ✅ **Graph visualization** for debugging and understanding
- ✅ **Real-world applications** and production patterns

### **Advanced Topics to Explore Next**

#### **1. Custom Runnables**
```python
from langchain_core.runnables import Runnable

class CustomProcessor(Runnable):
    def invoke(self, input_data):
        # Your custom logic here
        return processed_data
```

#### **2. Memory Integration**
```python
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory()
chain_with_memory = memory | your_chain
```

#### **3. Tool Integration**
```python
from langchain.agents import initialize_agent
from langchain.tools import Tool

tools = [Tool(name="calculator", func=calculate)]
agent_chain = initialize_agent(tools, llm, agent_type="zero-shot")
```

#### **4. Async Chains**
```python
async def async_chain_execution():
    result = await chain.ainvoke(input_data)
    return result
```

### **Production Deployment Checklist**

- [ ] **Error handling** implemented for all chains
- [ ] **Input validation** to prevent malformed data
- [ ] **Rate limiting** to prevent API abuse
- [ ] **Logging** for monitoring and debugging
- [ ] **Performance monitoring** for optimization
- [ ] **Testing suite** for regression prevention
- [ ] **Fallback mechanisms** for system reliability
- [ ] **Documentation** for maintenance and scaling

### **Resources for Continued Learning**

- **LangChain Documentation**: Deep dive into advanced features
- **Community Examples**: Real-world implementations
- **Performance Optimization**: Advanced techniques
- **Production Deployment**: Scaling and reliability patterns

---

## 📚 **Summary: Your Chain Mastery Journey**

### **Simple Chains** → Foundation
- **Concept**: Sequential processing with pipe operator
- **Use case**: Linear workflows, step-by-step transformations
- **Key skill**: Understanding component composition

### **Parallel Chains** → Optimization  
- **Concept**: Concurrent processing with RunnableParallel
- **Use case**: Independent operations, performance improvement
- **Key skill**: Resource optimization and result merging

### **Conditional Chains** → Intelligence
- **Concept**: Dynamic routing with RunnableBranch
- **Use case**: Adaptive behavior, content-based decisions
- **Key skill**: Logic implementation and structured output

### **Combined Patterns** → Production
- **Concept**: Using all three types together
- **Use case**: Complex, real-world applications
- **Key skill**: System architecture and error handling

You're now ready to build sophisticated AI applications using LangChain chains! Start with simple use cases and gradually incorporate more advanced patterns as your needs grow.

**Happy chaining! 🔗✨**
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
