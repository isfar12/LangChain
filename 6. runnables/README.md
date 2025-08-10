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

#### **Step 1: Import Essential Components**
```python
from langchain_core.runnables import RunnableSequence
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
```

**What each import does:**
- `RunnableSequence`: **The conductor** - orchestrates multiple operations in order
- `ChatOllama`: **The AI brain** - connects to your local Ollama models
- `PromptTemplate`: **The instruction creator** - builds dynamic prompts with placeholders
- `StrOutputParser`: **The text extractor** - converts complex model responses to clean strings

#### **Step 2: Configure the AI Model**
```python
model = ChatOllama(
    model="gemma3:1b",      # Lightweight, fast model (1 billion parameters)
    temperature=0.7         # Creativity level: 0=robotic, 1=very creative
)
```

**Model configuration explained:**
- `gemma3:1b`: Fast model good for simple tasks (trade-off: speed vs capability)
- `temperature=0.7`: Sweet spot for creative but coherent responses
  - `0.0-0.3`: Very focused, factual, repeatable
  - `0.4-0.7`: Balanced creativity and coherence ✅
  - `0.8-1.0`: Highly creative, potentially inconsistent

#### **Step 3: Design Prompt Templates**
```python
# Template 1: Joke Generation
prompt1 = PromptTemplate(
    template="Write me a joke about {topic}",
    input_variables=["topic"]
)

# Template 2: Joke Explanation 
prompt2 = PromptTemplate(
    template="Explain the following joke: {joke} \n First write the joke, then explain it.",
    input_variables=["joke"]
)
```

**Template design principles:**
- `{topic}` and `{joke}`: **Placeholders** that get replaced with actual values
- `input_variables`: **Required list** of all placeholders in the template
- **Clear instructions**: "Write me a joke" vs vague "tell me something funny"
- **Specific format**: "First write the joke, then explain it" guides output structure

#### **Step 4: Set Up Output Processing**
```python
parser = StrOutputParser()
```

**Why we need a parser:**
- AI models return complex response objects with metadata
- `StrOutputParser()` extracts just the text content we want
- **Input**: `AIMessage(content="Why did the cat...?", metadata={...})`
- **Output**: `"Why did the cat...?"`

#### **Step 5: Build the Sequential Chain**
```python
chain = RunnableSequence(prompt1, model, parser, prompt2, model, parser)
```

**Breaking down the sequence:**
1. `prompt1` → Takes `{"topic": "cats"}` → Creates "Write me a joke about cats"
2. `model` → Processes prompt → Generates joke about cats
3. `parser` → Extracts clean text → "Why did the cat cross the road? To get to the other side!"
4. `prompt2` → Uses joke text → Creates "Explain the following joke: Why did the cat..."
5. `model` → Processes explanation prompt → Generates joke explanation
6. `parser` → Extracts final clean text → Complete explanation

**Data transformation flow:**
```
Input: {"topic": "cats"}
   ↓ (prompt1 processes input)
Prompt: "Write me a joke about cats"  
   ↓ (model generates response)
Joke: "Why did the cat cross the road?..."
   ↓ (parser cleans response)
Clean text: "Why did the cat cross the road?..."
   ↓ (prompt2 uses joke as input)
Prompt: "Explain the following joke: Why did the cat..."
   ↓ (model generates explanation)
Explanation: "This joke is funny because..."
   ↓ (parser cleans final response)
Final output: "This joke is funny because..."
```

#### **Step 6: Execute and Test**
```python
result = chain.invoke({"topic": "cats"})
print(result)
```

**What happens during execution:**
- `invoke()`: Starts the chain with input data
- Each component processes data sequentially
- Returns final processed result

**Expected output structure:**
```
"Why did the cat cross the road? To get to the other side!

This joke is funny because it's a play on the classic 'Why did the chicken cross the road?' joke. By substituting a cat for a chicken, it creates a familiar yet unexpected twist that subverts the listener's expectations while maintaining the same simple punchline structure."
```

### **Chain Construction Patterns**

```python
# Pattern 1: Linear processing
chain = RunnableSequence(step1, step2, step3, step4)

# Pattern 2: Processing with different models  
chain = RunnableSequence(
    prompt1, model_a, parser,    # First AI operation
    prompt2, model_b, parser,    # Second AI operation with different model
    prompt3, model_a, parser     # Third operation back to first model
)

# Pattern 3: Multi-stage refinement
chain = RunnableSequence(
    draft_prompt, model, parser,        # Generate initial draft
    review_prompt, critic_model, parser,  # Review and critique
    revision_prompt, model, parser      # Revise based on feedback
)
```

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

#### **Step 1: Import Parallel Components**
```python
from langchain_core.runnables import RunnableParallel, RunnableSequence
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
```

**New import spotlight:**
- `RunnableParallel`: **The multitasker** - runs multiple operations simultaneously
- `RunnableSequence`: **Still needed** - for creating individual chains within parallel branches

#### **Step 2: Configure Multiple AI Models**
```python
# Model 1: Fast and efficient
model1 = ChatOllama(
    model="gemma3:1b",      # 1B parameters - quick responses
    temperature=0.7         # Balanced creativity
)

# Model 2: More sophisticated  
model2 = ChatOllama(
    model="gemma2:2b",      # 2B parameters - more nuanced responses
    temperature=0.7         # Same creativity level for consistency
)
```

**Multi-model strategy:**
- **Specialization**: Different models excel at different tasks
- **Load balancing**: Distribute work across models
- **Comparison**: Generate variations for A/B testing
- **Redundancy**: Backup if one model is slow or unavailable

**Performance comparison:**
- `gemma3:1b`: ⚡ Faster, lighter, good for simple tasks
- `gemma2:2b`: 🧠 Slower, more capable, better for complex tasks

#### **Step 3: Design Content-Specific Prompts**
```python
# LinkedIn prompt: Professional, longer format
prompt1 = PromptTemplate(
    template="Write me a Linkedin medium length post about {topic}, Just give me the post, no explanation.",
    input_variables=["topic"]
)

# Twitter prompt: Casual, short format
prompt2 = PromptTemplate(
    template="Write me a tweet about {topic}, just give me the tweet, no explanation.",
    input_variables=["topic"]
)
```

**Platform-specific design:**
- **LinkedIn**: "medium length post" → 150-300 words, professional tone
- **Twitter**: "tweet" → 280 characters max, casual, engaging
- **"Just give me the X, no explanation"**: Prevents unwanted meta-commentary

#### **Step 4: Build Individual Processing Chains**
```python
parser = StrOutputParser()

# Chain 1: LinkedIn content generation
linkedin_chain = RunnableSequence(prompt1, model1, parser)

# Chain 2: Twitter content generation  
twitter_chain = RunnableSequence(prompt2, model2, parser)
```

**Chain composition breakdown:**
- Each chain is **self-contained**: prompt → model → parser
- **Different models**: LinkedIn uses `model1`, Twitter uses `model2`
- **Same parser**: Both use `StrOutputParser` for text extraction

#### **Step 5: Create Parallel Execution Structure**
```python
parallel_chain = RunnableParallel({
    "linkedin_post": RunnableSequence(prompt1, model1, parser),
    "tweet": RunnableSequence(prompt2, model2, parser)
})
```

**Dictionary structure explained:**
```python
{
    "output_key": processing_chain,
    "another_key": another_chain,
    # Add as many parallel branches as needed
}
```

**What happens during parallel execution:**
1. **Input distribution**: Same input `{"topic": "AI"}` goes to both chains
2. **Simultaneous processing**: Both chains run at the same time
3. **Result collection**: Outputs collected in dictionary format
4. **Final structure**: `{"linkedin_post": "...", "tweet": "..."}`

#### **Step 6: Execute and Retrieve Results**
```python
result = parallel_chain.invoke({"topic": "AI and its impact on society"})

# Access individual results
print(result["linkedin_post"])   # LinkedIn post content
print(result["tweet"])           # Tweet content
```

**Result handling patterns:**
```python
# Pattern 1: Print both
print("LinkedIn:", result["linkedin_post"])
print("Twitter:", result["tweet"])

# Pattern 2: Save to variables
linkedin_content = result["linkedin_post"]  
tweet_content = result["tweet"]

# Pattern 3: Further processing
for platform, content in result.items():
    print(f"{platform}: {len(content)} characters")
```

### **Performance Benefits Deep Dive**

#### **Timing Comparison**
```python
import time

# Sequential execution (old way)
start_time = time.time()
linkedin_result = linkedin_chain.invoke({"topic": "AI"})
tweet_result = twitter_chain.invoke({"topic": "AI"}) 
sequential_time = time.time() - start_time

# Parallel execution (new way)
start_time = time.time()
parallel_result = parallel_chain.invoke({"topic": "AI"})
parallel_time = time.time() - start_time

print(f"Sequential: {sequential_time:.2f}s")
print(f"Parallel: {parallel_time:.2f}s") 
print(f"Speed improvement: {sequential_time/parallel_time:.2f}x")
```

**Expected performance:**
- Sequential: ~8-12 seconds (4-6s per model call)
- Parallel: ~4-6 seconds (models run simultaneously)
- **Speed improvement: ~2x faster!**

### Execution Flow

```bash
Input: {"topic": "AI and its impact on society"}
                    ↓
        ┌───── PARALLEL SPLIT ─────┐
        ↓                         ↓
"linkedin_post" branch    "tweet" branch
        ↓                         ↓
prompt1 → model1 → parser  prompt2 → model2 → parser
        ↓                         ↓
   LinkedIn content          Tweet content
        ↓                         ↓
        └───── PARALLEL JOIN ─────┘
                    ↓
Output: {
    "linkedin_post": "AI is revolutionizing...",
    "tweet": "🤖 AI is changing everything! #AI #Tech"
}
```
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

#### **Step 1: Import Passthrough Components**
```python
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableSequence
```

**New component spotlight:**
- `RunnablePassthrough`: **The preserver** - passes input through unchanged
- Think of it as a "photocopy" that keeps the original while processing continues

#### **Step 2: Build Initial Content Generator**
```python
# Initial content creation chain
linkedin_generator = RunnableSequence(prompt1, model1, parser)
```

**What this chain does:**
1. Takes topic input: `{"topic": "AI"}`
2. Creates LinkedIn post about AI
3. Returns clean text: `"AI is transforming industries..."`

#### **Step 3: Design Parallel Processing with Passthrough**
```python
parallel_chain = RunnableParallel({
    "linkedin_post": RunnablePassthrough(),               # Preserves original
    "summary": RunnableSequence(prompt2, model2, parser)  # Creates summary
})
```

**Key insight - Data flow:**
```python
# Input to parallel_chain: "AI is transforming industries by..."
#                         ↓
# ┌─────────────────────────────┬──────────────────────────────────┐
# ↓                             ↓                                  ↓
# "linkedin_post":              "summary":
# RunnablePassthrough()         RunnableSequence(prompt2, model2, parser)
# ↓                             ↓
# "AI is transforming..."       "This post discusses AI's impact..."
# (original, unchanged)         (newly generated summary)
```

**Why use passthrough?**
- **Preserve original**: Keep the full LinkedIn post intact
- **Enable comparison**: Original vs processed versions
- **Maintain context**: Sometimes you need both versions
- **Efficiency**: No re-processing of already good content

#### **Step 4: Create Complete Processing Pipeline**
```python
final_chain = RunnableSequence(linkedin_generator, parallel_chain)
```

**Complete flow breakdown:**
```
Step 1: linkedin_generator
Input: {"topic": "AI and its impact on society"}
   ↓
Output: "AI is revolutionizing industries across the globe..."

Step 2: parallel_chain  
Input: "AI is revolutionizing industries across the globe..."
   ↓
Parallel processing:
├─ "linkedin_post": RunnablePassthrough() → original post
└─ "summary": prompt2 → model2 → parser → summary

Final Output: {
    "linkedin_post": "AI is revolutionizing industries...",
    "summary": "This post explores AI's transformative impact..."
}
```

#### **Step 5: Advanced Passthrough Patterns**
```python
# Pattern 1: Preserve + Multiple Processing
advanced_parallel = RunnableParallel({
    "original": RunnablePassthrough(),
    "summary": summary_chain,
    "sentiment": sentiment_chain,
    "keywords": keyword_extraction_chain
})

# Pattern 2: Selective Preservation
conditional_passthrough = RunnableParallel({
    "content": RunnablePassthrough(),
    "processed": RunnableBranch(
        (lambda x: len(x) > 500, summary_chain),  # Summarize if too long
        RunnablePassthrough()                      # Pass through if short enough
    )
})

# Pattern 3: Metadata Addition
metadata_parallel = RunnableParallel({
    "content": RunnablePassthrough(),
    "word_count": RunnableLambda(lambda x: len(x.split())),
    "reading_time": RunnableLambda(lambda x: f"{len(x.split()) // 200 + 1} min read")
})
```

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

### Code Explanation

#### **Step 1: Import Lambda Components**
```python
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnableSequence
```

**New component spotlight:**
- `RunnableLambda`: **The customizer** - wraps any Python function to work in chains
- Bridges the gap between LangChain components and your custom logic

#### **Step 2: Define Custom Processing Functions**
```python
def count_words(text):
    """Custom function to count words in text"""
    return len(text.split())

# Alternative implementations:
def count_words_detailed(text):
    """More detailed word counting with metadata"""
    words = text.split()
    return {
        "word_count": len(words),
        "character_count": len(text),
        "sentence_count": len(text.split('.')),
        "avg_word_length": sum(len(word) for word in words) / len(words) if words else 0
    }

def extract_hashtags(text):
    """Extract potential hashtags from content"""
    import re
    # Simple hashtag extraction (in real implementation, use NLP)
    words = re.findall(r'\b[A-Z][a-zA-Z]+\b', text)  # Find capitalized words
    return [f"#{word.lower()}" for word in words[:5]]  # Return first 5 as hashtags
```

**Custom function requirements:**
- **Input**: Must accept the data passed from previous step
- **Output**: Must return data that next step can process  
- **Pure function**: No side effects (don't modify global variables)
- **Error handling**: Handle edge cases gracefully

#### **Step 3: Build Content Generation Chain**
```python
linkedin_generator = RunnableSequence(prompt1, chat, parser)
```

**This creates our base content** that will be analyzed by custom functions

#### **Step 4: Integrate Custom Functions with RunnableLambda**
```python
parallel_chain = RunnableParallel({
    "linkedin_post": linkedin_generator,
    "word_count": RunnableLambda(count_words)
})
```

**Lambda integration breakdown:**
- `RunnableLambda(count_words)`: Wraps the function to work in chains
- **Input to lambda**: Receives the LinkedIn post text
- **Function execution**: `count_words("AI is transforming...")` 
- **Output**: Returns word count as integer

#### **Step 5: Create Complete Analysis Pipeline**
```python
final_chain = RunnableSequence(linkedin_generator, parallel_chain)
```

**Data flow with custom function:**
```
Input: {"topic": "AI and its impact on society"}
   ↓
linkedin_generator: "AI is transforming industries by automating..."
   ↓
parallel_chain splits to:
├─ "linkedin_post": linkedin_generator → "AI is transforming..."
└─ "word_count": RunnableLambda(count_words) → 156

Final Output: {
    "linkedin_post": "AI is transforming industries by automating...",
    "word_count": 156
}
```

#### **Step 6: Advanced Lambda Patterns**
```python
# Pattern 1: Complex analysis chain
analysis_parallel = RunnableParallel({
    "content": RunnablePassthrough(),
    "word_count": RunnableLambda(count_words),
    "readability": RunnableLambda(lambda x: "Easy" if len(x.split()) < 100 else "Complex"),
    "hashtags": RunnableLambda(extract_hashtags),
    "sentiment": RunnableLambda(lambda x: "Positive" if "great" in x.lower() else "Neutral")
})

# Pattern 2: Data transformation chain  
transform_chain = RunnableSequence(
    content_generator,
    RunnableLambda(lambda x: x.upper()),  # Convert to uppercase
    RunnableLambda(lambda x: x.replace("AI", "Artificial Intelligence")),  # Expand acronyms
    RunnableLambda(lambda x: f"ANNOUNCEMENT: {x}")  # Add prefix
)

# Pattern 3: Validation and filtering
validation_chain = RunnableSequence(
    content_generator,
    RunnableBranch(
        (lambda x: len(x.split()) < 50, RunnableLambda(lambda x: "Content too short")),
        (lambda x: len(x.split()) > 300, RunnableLambda(lambda x: "Content too long")), 
        RunnablePassthrough()  # Content is just right
    )
)
```

#### **Step 7: Error Handling in Custom Functions**
```python
def safe_word_count(text):
    """Word counting with error handling"""
    try:
        if not isinstance(text, str):
            return 0
        if not text.strip():
            return 0
        return len(text.split())
    except Exception as e:
        print(f"Error counting words: {e}")
        return 0

# Use in chain with error handling
safe_chain = RunnableParallel({
    "content": linkedin_generator,
    "word_count": RunnableLambda(safe_word_count)
})
```

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

### **Real-World Lambda Applications**

#### **Content Quality Assessment**
```python
def assess_content_quality(text):
    """Comprehensive content quality analysis"""
    words = text.split()
    sentences = text.split('.')
    
    quality_score = 0
    feedback = []
    
    # Word count check
    if 50 <= len(words) <= 300:
        quality_score += 25
        feedback.append("✅ Good length")
    else:
        feedback.append("⚠️ Consider adjusting length")
    
    # Sentence variety check
    avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0
    if 10 <= avg_sentence_length <= 20:
        quality_score += 25
        feedback.append("✅ Good sentence variety")
    
    # Engagement elements
    if any(char in text for char in '!?'):
        quality_score += 25
        feedback.append("✅ Engaging punctuation")
    
    # Call-to-action check
    cta_words = ['share', 'comment', 'like', 'follow', 'subscribe', 'join']
    if any(word.lower() in text.lower() for word in cta_words):
        quality_score += 25
        feedback.append("✅ Includes call-to-action")
    
    return {
        "score": quality_score,
        "grade": "A" if quality_score >= 75 else "B" if quality_score >= 50 else "C",
        "feedback": feedback
    }

# Use in content creation pipeline
quality_chain = RunnableParallel({
    "content": content_generator,
    "quality_analysis": RunnableLambda(assess_content_quality)
})
```

#### **Social Media Optimization**
```python
def optimize_for_platform(content):
    """Optimize content for different social media platforms"""
    
    def create_twitter_version(text):
        # Truncate and add hashtags for Twitter
        words = text.split()[:30]  # Roughly 280 characters
        truncated = ' '.join(words)
        if len(truncated) < len(text):
            truncated += "... 🧵"
        return truncated
    
    def create_instagram_version(text):
        # Add emoji and hashtags for Instagram
        emojis = ["🚀", "💡", "🌟", "✨", "🔥"]
        lines = text.split('.')[:3]  # First 3 sentences
        formatted = ' '.join(lines) + f" {emojis[0]}"
        return formatted
    
    return {
        "original": content,
        "twitter": create_twitter_version(content),
        "instagram": create_instagram_version(content),
        "linkedin": content  # Keep original for LinkedIn
    }

# Multi-platform content chain
platform_optimizer = RunnableSequence(
    base_content_generator,
    RunnableLambda(optimize_for_platform)
)
```

### Common Use Cases for RunnableLambda

#### **1. Data Transformation and Cleaning**
```python
# Text cleaning
clean_text = RunnableLambda(lambda x: x.strip().replace('\n', ' '))

# Format standardization  
format_date = RunnableLambda(lambda x: datetime.strptime(x, "%Y-%m-%d").strftime("%B %d, %Y"))

# Data validation
validate_email = RunnableLambda(lambda x: x if "@" in x and "." in x else "invalid_email")
```

#### **2. Custom Calculations and Metrics**
```python
# Readability scoring
readability_score = RunnableLambda(lambda x: len(x.split()) / len(x.split('.')))

# Sentiment scoring (simple version)
simple_sentiment = RunnableLambda(lambda x: sum(1 for word in ['great', 'awesome', 'excellent'] if word in x.lower()))

# SEO metrics
seo_analysis = RunnableLambda(lambda x: {
    "word_count": len(x.split()),
    "keyword_density": x.lower().count("ai") / len(x.split()) * 100,
    "has_question": "?" in x
})
```

#### **3. External API Integration**
```python
import requests

def translate_text(text, target_language="es"):
    """Translate text using external API (mock example)"""
    try:
        # In real implementation, use Google Translate API or similar
        api_response = requests.post("https://api.translate.service.com", {
            "text": text,
            "target": target_language
        })
        return api_response.json()["translated_text"]
    except:
        return f"Translation failed for: {text[:50]}..."

# Translation chain
translation_chain = RunnableParallel({
    "english": RunnablePassthrough(),
    "spanish": RunnableLambda(lambda x: translate_text(x, "es")),
    "french": RunnableLambda(lambda x: translate_text(x, "fr"))
})
```

#### **4. File Operations and Storage**
```python
import json
from datetime import datetime

def save_content(content):
    """Save content to file with timestamp"""
    timestamp = datetime.now().isoformat()
    filename = f"content_{timestamp}.txt"
    
    with open(filename, 'w') as f:
        f.write(content)
    
    return {
        "content": content,
        "saved_to": filename,
        "timestamp": timestamp
    }

# Content saving chain
save_chain = RunnableSequence(
    content_generator,
    RunnableLambda(save_content)
)
```

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

## **5. RunnableBranch: Intelligent Decision Making**

### **Code Breakdown: `runnable_branch.py`**

```python
# Step 1: Environment Setup and Imports
from langchain_core.runnables import RunnableBranch, RunnableLambda, RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama

# These imports provide:
# - RunnableBranch: For conditional logic and decision making
# - Supporting Runnables for data flow control
# - Prompt management for different content types
# - Local model integration with ChatOllama

# Step 2: Model Configuration
llm = ChatOllama(
    model="llama3.1",
    temperature=0.1,
    max_tokens=150
)

# Low temperature (0.1) ensures consistent, predictable responses
# Token limit controls response length for different content types
```

### **Understanding RunnableBranch Decision Logic**

#### **How Conditional Routing Works**

```python
# Step 3: Define Content Type Detection Functions
def is_technical_request(user_input):
    """
    Analyzes user input to determine if it's technical content
    
    Detection Strategy:
    - Keyword matching for technical terms
    - Context analysis for programming/tech concepts
    - Domain-specific vocabulary identification
    """
    technical_keywords = [
        'code', 'programming', 'algorithm', 'database',
        'API', 'function', 'class', 'variable', 'loop',
        'framework', 'library', 'debugging', 'deployment'
    ]
    
    user_lower = user_input.lower()
    technical_count = sum(1 for keyword in technical_keywords if keyword in user_lower)
    
    # Decision threshold: 2+ technical terms = technical content
    return technical_count >= 2

def is_creative_request(user_input):
    """
    Identifies creative writing requests
    
    Creative Indicators:
    - Artistic and expressive language
    - Storytelling elements
    - Emotional or imaginative content
    """
    creative_keywords = [
        'story', 'poem', 'creative', 'imagine', 'character',
        'plot', 'narrative', 'artistic', 'emotional', 'inspire',
        'dream', 'fantasy', 'adventure', 'romance', 'mystery'
    ]
    
    user_lower = user_input.lower()
    creative_count = sum(1 for keyword in creative_keywords if keyword in user_lower)
    
    return creative_count >= 1

# Step 4: Create Specialized Prompt Templates
technical_prompt = PromptTemplate(
    input_variables=["input"],
    template="""
    📋 TECHNICAL RESPONSE MODE ACTIVATED
    
    User Request: {input}
    
    Please provide a clear, structured technical response with:
    
    1. **Core Concept**: Define key technical terms
    2. **Implementation**: Step-by-step approach
    3. **Best Practices**: Industry standards and recommendations
    4. **Common Pitfalls**: What to avoid
    5. **Resources**: Further learning materials
    
    Format your response with clear headings and bullet points.
    Focus on accuracy and practical applicability.
    """
)

creative_prompt = PromptTemplate(
    input_variables=["input"],
    template="""
    🎨 CREATIVE RESPONSE MODE ACTIVATED
    
    User Request: {input}
    
    Let your imagination flow! Create engaging content with:
    
    ✨ Rich, descriptive language
    ✨ Emotional depth and connection  
    ✨ Vivid imagery and metaphors
    ✨ Compelling narrative structure
    ✨ Unique perspective and voice
    
    Make it memorable, inspiring, and emotionally resonant.
    Use creative formatting and expressive elements.
    """
)

general_prompt = PromptTemplate(
    input_variables=["input"],
    template="""
    💬 GENERAL RESPONSE MODE
    
    User Request: {input}
    
    Provide a balanced, informative response that:
    
    • Addresses the question directly
    • Offers practical insights
    • Uses clear, accessible language
    • Includes relevant examples
    • Suggests next steps if applicable
    
    Keep the tone professional yet approachable.
    """
)
```

### **Branch Construction and Decision Tree**

```python
# Step 5: Build the Decision Branch System
content_router = RunnableBranch(
    # Branch 1: Technical Content Route
    (
        # Condition: Check if input is technical
        lambda x: is_technical_request(x["input"]), 
        
        # Action: Apply technical prompt + model processing
        technical_prompt | llm
    ),
    
    # Branch 2: Creative Content Route  
    (
        # Condition: Check if input is creative
        lambda x: is_creative_request(x["input"]),
        
        # Action: Apply creative prompt + model processing
        creative_prompt | llm
    ),
    
    # Branch 3: Default Route (fallback)
    # No condition needed - this catches everything else
    general_prompt | llm
)

# Decision Flow Visualization:
#
# User Input
#      |
#      v
# Is Technical? ——— YES ——→ Technical Prompt ——→ LLM ——→ Technical Response
#      |
#      NO
#      |
#      v  
# Is Creative? ——— YES ——→ Creative Prompt ——→ LLM ——→ Creative Response
#      |
#      NO
#      |
#      v
# Default Route ——————————→ General Prompt ——→ LLM ——→ General Response
```

### **Advanced Branching Patterns**

#### **Multi-Level Decision Trees**
```python
# Complex branching with nested conditions
advanced_router = RunnableBranch(
    # Primary branch: Technical content
    (
        lambda x: is_technical_request(x["input"]),
        RunnableBranch(
            # Sub-branch: Programming vs Infrastructure  
            (lambda x: any(word in x["input"].lower() for word in ['code', 'programming', 'function']), programming_prompt | llm),
            (lambda x: any(word in x["input"].lower() for word in ['server', 'deployment', 'cloud']), infrastructure_prompt | llm),
            # Default technical fallback
            technical_prompt | llm
        )
    ),
    
    # Primary branch: Creative content
    (
        lambda x: is_creative_request(x["input"]),
        RunnableBranch(
            # Sub-branch: Fiction vs Poetry
            (lambda x: any(word in x["input"].lower() for word in ['story', 'character', 'plot']), fiction_prompt | llm),
            (lambda x: any(word in x["input"].lower() for word in ['poem', 'verse', 'rhyme']), poetry_prompt | llm),
            # Default creative fallback
            creative_prompt | llm
        )
    ),
    
    # Default fallback
    general_prompt | llm
)
```

#### **Production-Ready Branch Examples**

```python
def classify_support_request(message):
    """Classify customer support requests by urgency and type"""
    urgent_keywords = ['urgent', 'critical', 'down', 'error', 'broken', 'not working']
    billing_keywords = ['payment', 'invoice', 'charge', 'billing', 'refund']
    technical_keywords = ['api', 'integration', 'code', 'authentication', 'ssl']
    
    is_urgent = any(keyword in message.lower() for keyword in urgent_keywords)
    is_billing = any(keyword in message.lower() for keyword in billing_keywords)
    is_technical = any(keyword in message.lower() for keyword in technical_keywords)
    
    return {
        'urgent': is_urgent,
        'billing': is_billing,
        'technical': is_technical,
        'priority': 'high' if is_urgent else 'medium' if is_technical or is_billing else 'low'
    }

support_router = RunnableBranch(
    # High priority - immediate response
    (
        lambda x: classify_support_request(x["message"])['urgent'],
        urgent_support_prompt | llm
    ),
    # Billing issues - specialized handling
    (
        lambda x: classify_support_request(x["message"])['billing'],
        billing_support_prompt | llm  
    ),
    # Technical issues - detailed troubleshooting
    (
        lambda x: classify_support_request(x["message"])['technical'],
        technical_support_prompt | llm
    ),
    # General inquiries
    general_support_prompt | llm
)
```

### **RunnableBranch Best Practices**

#### **1. Condition Design Guidelines**
```python
# ✅ Good: Clear, testable conditions
def is_question(text):
    return text.strip().endswith('?') or text.lower().startswith(('what', 'how', 'why', 'when', 'where'))

# ❌ Avoid: Complex, hard-to-test conditions  
lambda x: len(x) > 10 and 'help' in x and not x.startswith('no') and x.count(' ') > 3
```

#### **2. Error Handling in Branches**
```python
def safe_condition(check_function):
    """Wrapper for safe condition checking"""
    def wrapper(x):
        try:
            return check_function(x)
        except Exception as e:
            print(f"Condition check failed: {e}")
            return False
    return wrapper

# Use safe conditions
safe_router = RunnableBranch(
    (safe_condition(is_technical_request), technical_prompt | llm),
    (safe_condition(is_creative_request), creative_prompt | llm),
    general_prompt | llm
)
```

---

## **Quick Reference Summary**

### **Runnable Types Overview**

### 1. RunnableSequence
- **Purpose**: Sequential processing pipeline
- **Use Case**: Step-by-step transformations
- **Pattern**: Input → Step 1 → Step 2 → Step 3 → Output

### 2. RunnableParallel
- **Purpose**: Concurrent processing with multiple outputs  
- **Use Case**: Multi-model strategies, parallel analysis
- **Pattern**: Input → [Path A, Path B, Path C] → Combined Output

### 3. RunnablePassthrough
- **Purpose**: Data preservation and selective processing
- **Use Case**: Maintaining original data while adding analysis
- **Pattern**: Input → {original: Input, processed: Transform(Input)}

### 4. RunnableLambda
- **Purpose**: Integrate custom Python functions
- **Use Case**: Custom processing, calculations, transformations
- **Pattern**: Input → Custom Function → Output

### 5. RunnableBranch
- **Purpose**: Conditional execution based on criteria
- **Use Case**: Different processing paths based on content analysis  
- **Pattern**: Input → Condition Check → Path A or Path B

---

## **Advanced Integration Patterns**

### **Combining All Runnable Types**

```python
# Comprehensive workflow example
def build_comprehensive_content_pipeline():
    """
    Creates a sophisticated content processing pipeline that combines
    all Runnable types for maximum flexibility and power
    """
    
    # Step 1: Input preprocessing with Lambda
    input_cleaner = RunnableLambda(lambda x: {
        "original": x,
        "cleaned": x.strip().lower(),
        "word_count": len(x.split()),
        "timestamp": datetime.now().isoformat()
    })
    
    # Step 2: Multi-model parallel analysis
    parallel_analysis = RunnableParallel({
        # Preserve cleaned input
        "input_data": RunnablePassthrough(),
        
        # Content classification
        "classification": RunnableLambda(classify_content_type),
        
        # Sentiment analysis
        "sentiment": RunnableLambda(analyze_sentiment),
        
        # SEO metrics
        "seo_metrics": RunnableLambda(calculate_seo_metrics),
        
        # Readability score
        "readability": RunnableLambda(calculate_readability)
    })
    
    # Step 3: Conditional content generation based on classification
    content_generator = RunnableBranch(
        # Technical content route
        (
            lambda x: x["classification"]["type"] == "technical",
            RunnableSequence(
                technical_prompt,
                llm,
                RunnableLambda(add_technical_formatting)
            )
        ),
        
        # Marketing content route  
        (
            lambda x: x["classification"]["type"] == "marketing",
            RunnableParallel({
                "short_form": marketing_short_prompt | llm,
                "long_form": marketing_long_prompt | llm,
                "social_media": social_media_prompt | llm
            })
        ),
        
        # Creative content route
        (
            lambda x: x["classification"]["type"] == "creative",
            RunnableSequence(
                creative_prompt,
                llm,
                RunnableLambda(add_creative_formatting)
            )
        ),
        
        # Default general content
        general_prompt | llm
    )
    
    # Step 4: Post-processing and quality assurance
    quality_processor = RunnableParallel({
        "content": RunnablePassthrough(),
        "quality_check": RunnableLambda(quality_assessment),
        "optimization_suggestions": RunnableLambda(get_optimization_tips),
        "final_metrics": RunnableLambda(calculate_final_metrics)
    })
    
    # Step 5: Final output formatting
    output_formatter = RunnableLambda(format_final_output)
    
    # Combine everything into the complete pipeline
    complete_pipeline = RunnableSequence(
        input_cleaner,           # Clean and prepare input
        parallel_analysis,       # Analyze from multiple angles
        content_generator,       # Generate content based on type
        quality_processor,       # Quality check and optimization
        output_formatter         # Format final output
    )
    
    return complete_pipeline

# Usage example
def demonstrate_complete_pipeline():
    """Demonstrate the complete content pipeline"""
    
    pipeline = build_comprehensive_content_pipeline()
    
    # Test with different input types
    test_inputs = [
        "Write a technical guide on implementing microservices architecture",
        "Create a marketing campaign for a new eco-friendly product",
        "Tell me a story about a robot learning to feel emotions",
        "What are the best practices for remote team management?"
    ]
    
    for input_text in test_inputs:
        print(f"\n{'='*60}")
        print(f"Processing: {input_text[:50]}...")
        print('='*60)
        
        result = pipeline.invoke(input_text)
        
        print(f"Content Type: {result.get('classification', {}).get('type', 'Unknown')}")
        print(f"Quality Score: {result.get('quality_check', {}).get('score', 'N/A')}")
        print(f"Word Count: {result.get('final_metrics', {}).get('word_count', 'N/A')}")
        print(f"\nGenerated Content:\n{result.get('content', 'No content generated')[:200]}...")
```

### **Error Handling and Resilience Patterns**

```python
def create_resilient_pipeline():
    """
    Creates a pipeline with comprehensive error handling
    and fallback mechanisms
    """
    
    # Safe wrapper for any Runnable
    def safe_runnable(runnable, fallback_value=None, error_message="Processing failed"):
        def safe_wrapper(x):
            try:
                return runnable.invoke(x)
            except Exception as e:
                print(f"{error_message}: {str(e)}")
                return fallback_value or {"error": str(e), "fallback": True}
        
        return RunnableLambda(safe_wrapper)
    
    # Resilient pipeline with fallbacks
    resilient_pipeline = RunnableSequence(
        # Safe input processing
        safe_runnable(
            RunnableLambda(process_input),
            fallback_value={"processed": False, "raw_input": True},
            error_message="Input processing failed"
        ),
        
        # Safe parallel processing with individual fallbacks
        RunnableParallel({
            "primary": safe_runnable(
                primary_processor,
                fallback_value={"status": "fallback", "method": "basic"}
            ),
            "secondary": safe_runnable(
                secondary_processor, 
                fallback_value={"status": "unavailable"}
            ),
            "backup": RunnableLambda(basic_fallback_processor)  # Always works
        }),
        
        # Safe conditional processing
        RunnableBranch(
            # Try advanced processing first
            (
                lambda x: x.get("primary", {}).get("status") != "fallback",
                safe_runnable(advanced_processor)
            ),
            # Fallback to basic processing
            safe_runnable(
                basic_processor,
                fallback_value={"content": "Basic response generated", "method": "fallback"}
            )
        )
    )
    
    return resilient_pipeline
```

### **Performance Optimization Strategies**

```python
# Caching for expensive operations
from functools import lru_cache
import asyncio

class OptimizedPipeline:
    def __init__(self):
        self.cache = {}
        
    @lru_cache(maxsize=1000)
    def cached_analysis(self, input_hash):
        """Cache expensive analysis operations"""
        # Expensive analysis logic here
        return analysis_result
    
    def create_optimized_pipeline(self):
        """Create a performance-optimized pipeline"""
        
        # Fast input preprocessing
        fast_preprocessor = RunnableLambda(lambda x: {
            "input": x,
            "hash": hash(x),
            "length": len(x),
            "type_hint": self.quick_type_detection(x)
        })
        
        # Intelligent caching layer
        cache_layer = RunnableLambda(lambda x: {
            **x,
            "cached_analysis": self.get_or_compute_analysis(x["hash"], x["input"])
        })
        
        # Parallel processing with load balancing
        balanced_parallel = RunnableParallel({
            "fast_path": RunnableLambda(self.fast_processing),
            "detailed_path": RunnableBranch(
                # Only do expensive processing if needed
                (lambda x: x["length"] > 100, expensive_processor),
                simple_processor
            ),
            "metadata": RunnablePassthrough()
        })
        
        # Optimized pipeline
        return RunnableSequence(
            fast_preprocessor,
            cache_layer,
            balanced_parallel,
            RunnableLambda(self.merge_results)
        )
    
    def quick_type_detection(self, text):
        """Fast heuristic type detection"""
        # Simple rule-based detection for speed
        if len(text.split()) < 10:
            return "short"
        elif any(word in text.lower() for word in ['code', 'function', 'api']):
            return "technical"
        else:
            return "general"
    
    def get_or_compute_analysis(self, input_hash, text):
        """Get cached analysis or compute new one"""
        if input_hash in self.cache:
            return self.cache[input_hash]
        
        result = self.cached_analysis(input_hash)
        self.cache[input_hash] = result
        return result
```

---

## **Best Practices for Production**

### **1. Error Handling Guidelines**
```python
# ✅ Always wrap risky operations
def safe_processor(data):
    try:
        return expensive_operation(data)
    except Exception as e:
        return {"error": str(e), "fallback": True}

# ✅ Use validation chains
validation_chain = RunnableSequence(
    RunnableLambda(validate_input),
    main_processor,
    RunnableLambda(validate_output)
)
```

### **2. Performance Optimization**
```python
# ✅ Use parallel execution for independent operations
parallel_chain = RunnableParallel({
    "analysis_a": processor_a,  # Independent
    "analysis_b": processor_b,  # Independent  
    "analysis_c": processor_c   # Independent
})

# ✅ Cache expensive operations
@lru_cache(maxsize=100)
def expensive_analysis(text_hash):
    return complex_computation(text_hash)
```

### **3. Modularity and Reusability**
```python
# ✅ Create reusable components
def create_content_analyzer(model_name, temperature=0.7):
    """Factory function for content analyzers"""
    model = ChatOllama(model=model_name, temperature=temperature)
    return RunnableSequence(
        content_prompt,
        model,
        StrOutputParser()
    )

# Use across different pipelines
technical_analyzer = create_content_analyzer("llama3.1", 0.1)
creative_analyzer = create_content_analyzer("gemma2:2b", 0.9)
```

### **4. Testing Strategies**
```python
# ✅ Unit test individual components
def test_content_classifier():
    classifier = RunnableLambda(classify_content)
    
    assert classifier.invoke("Python code example")["type"] == "technical"
    assert classifier.invoke("Once upon a time...")["type"] == "creative"
    assert classifier.invoke("How to invest money")["type"] == "general"

# ✅ Integration testing
def test_complete_pipeline():
    pipeline = create_content_pipeline()
    result = pipeline.invoke("Write a story about AI")
    
    assert "content" in result
    assert len(result["content"]) > 50
    assert result["classification"]["type"] == "creative"
```

### **5. Documentation Standards**
```python
def create_marketing_pipeline():
    """
    Creates a comprehensive marketing content pipeline.
    
    Pipeline Flow:
    1. Input validation and cleaning
    2. Audience analysis and segmentation  
    3. Parallel content generation for multiple channels
    4. Quality assessment and optimization
    5. Final formatting and delivery
    
    Args:
        None
    
    Returns:
        RunnableSequence: Complete marketing pipeline
        
    Example:
        >>> pipeline = create_marketing_pipeline()
        >>> result = pipeline.invoke("Launch campaign for eco-friendly products")
        >>> print(result["content"]["social_media"])
    """
    # Implementation here...
```

---

## **Troubleshooting Common Issues**

### **Memory and Performance Issues**
```python
# Problem: Pipeline consuming too much memory
# Solution: Use streaming and chunking
def create_memory_efficient_pipeline():
    return RunnableSequence(
        RunnableLambda(chunk_input),        # Break large inputs
        RunnableParallel({                  # Process chunks in parallel
            "chunk_1": process_chunk,
            "chunk_2": process_chunk,
            "chunk_3": process_chunk
        }),
        RunnableLambda(merge_chunks)        # Combine results
    )
```

### **Error Propagation Issues**
```python
# Problem: Errors stopping entire pipeline
# Solution: Graceful degradation
def create_fault_tolerant_pipeline():
    return RunnableBranch(
        # Try advanced processing
        (
            lambda x: advanced_processor_available(),
            RunnableSequence(
                advanced_preprocessor,
                advanced_model,
                advanced_postprocessor
            )
        ),
        # Fallback to basic processing
        RunnableSequence(
            basic_preprocessor,
            basic_model,
            basic_postprocessor
        )
    )
```

### **Debugging Complex Pipelines**
```python
# Add logging to track data flow
def debug_wrapper(step_name):
    def wrapper(x):
        print(f"🔍 Debug {step_name}: Input type={type(x)}, keys={list(x.keys()) if isinstance(x, dict) else 'N/A'}")
        return x
    return RunnableLambda(wrapper)

# Use in pipeline
debug_pipeline = RunnableSequence(
    debug_wrapper("start"),
    preprocessor,
    debug_wrapper("after_preprocess"),
    main_processor,
    debug_wrapper("after_main"),
    postprocessor,
    debug_wrapper("end")
)
```

---

## **Real-World Examples and Use Cases**

### **Content Management System**
```python
def build_cms_pipeline():
    """Complete CMS content processing pipeline"""
    return RunnableSequence(
        # Input validation
        RunnableLambda(validate_cms_input),
        
        # Parallel processing
        RunnableParallel({
            "content": RunnableBranch(
                (lambda x: x["type"] == "blog", blog_processor),
                (lambda x: x["type"] == "product", product_processor),
                (lambda x: x["type"] == "news", news_processor),
                default_content_processor
            ),
            "seo": seo_optimizer,
            "metadata": metadata_extractor,
            "original": RunnablePassthrough()
        }),
        
        # Final assembly
        RunnableLambda(assemble_cms_content)
    )
```

### **Customer Support Automation**
```python
def build_support_pipeline():
    """Automated customer support response system"""
    return RunnableSequence(
        # Ticket analysis
        RunnableParallel({
            "classification": ticket_classifier,
            "urgency": urgency_detector,
            "sentiment": sentiment_analyzer,
            "intent": intent_recognizer
        }),
        
        # Route to appropriate handler
        RunnableBranch(
            (lambda x: x["urgency"] == "critical", critical_handler),
            (lambda x: x["classification"] == "technical", technical_handler),
            (lambda x: x["classification"] == "billing", billing_handler),
            general_support_handler
        ),
        
        # Response formatting
        RunnableLambda(format_support_response)
    )
```

---

This comprehensive README provides a complete guide to understanding and implementing Runnables in LangChain, from basic concepts to advanced production patterns. Each example builds upon previous concepts, creating a clear learning path for mastering these powerful tools.
