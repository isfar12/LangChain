# LangChain Prompts - Comprehensive Tutorial

This directory contains a progressive series of examples demonstrating various prompt engineering techniques, message handling, and conversation management in LangChain. Each file builds upon previous concepts to create increasingly sophisticated chat applications.

## 📁 Directory Structure

```
langchain_prompts/
├── 1.basic_prompt_ui.py              # Streamlit UI with basic chat
├── 2.dynamic_prompt_basic.py         # Dynamic prompt generation
├── 3.basic_chatbot.py                # Command-line chatbot
├── 4.template_store_use/             # Template persistence
│   ├── template_gen.py               # Save templates to JSON
│   ├── new_prompt_ui.py              # Load templates from JSON
│   └── dynamic_prompt_template.json  # Stored template
├── 5.basics_of_messages.py           # Message types tutorial
├── 6.build_memory_in_chat/          # Conversation memory
│   └── modified_bot.py               # Chatbot with history
├── 7.dynamic_message_chatprompt.py  # ChatPromptTemplate usage
└── 8.dynamic_chat_history/          # Advanced memory management
    ├── dynamic_chat_history.py      # MessagesPlaceholder demo
    └── chat_history.txt              # Stored conversation
```

## 🚀 Tutorial Progression

### 1. **Basic Prompt UI** (`1.basic_prompt_ui.py`)

**What it does:**
- Creates a simple Streamlit web interface for chat interactions
- Uses basic system and human messages for structured conversations
- Provides immediate response without conversation memory

**Code breakdown:**

**Step 1: Import and Model Setup**
```python
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
import streamlit as st

chat = ChatOllama(model="phi3", temperature=0.7)
```
- `ChatOllama`: Creates a connection to the local Ollama service
- `temperature=0.7`: Controls creativity (0=deterministic, 1=very creative)
- Message imports: We'll use these to structure our conversation

**Step 2: Streamlit UI Components**
```python
st.header("Chat with Ollama Model")
user_input = st.text_input("Enter your message: ")
```
- `st.header()`: Creates a page title
- `st.text_input()`: Creates an input box for user messages

**Step 3: Message Processing and Response**
```python
if st.button("Send"):
    if user_input:
        messages=[
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content=user_input)
        ]
        response = chat.invoke(messages)
        st.write(f"Response: {response.content}")
```
- `SystemMessage`: Sets the AI's role/behavior (not visible to user)
- `HumanMessage`: Contains the actual user input
- `chat.invoke()`: Sends the message list to the model
- `response.content`: Extracts just the text from the AI response

**Key concepts learned:**
- Basic Streamlit UI creation
- SystemMessage for setting AI behavior/role
- HumanMessage for user input
- Single-turn conversations (no memory)

**Usage:** Run with `streamlit run 1.basic_prompt_ui.py`

---

### 2. **Dynamic Prompt Basic** (`2.dynamic_prompt_basic.py`)

**What it does:**
- Creates customizable prompts based on user selections
- Uses PromptTemplate for variable substitution
- Demonstrates dynamic content generation with structured options

**Code breakdown:**

**Step 1: User Interface Setup**
```python
st.header("Dynamic Prompt Customization")
paper_name = st.selectbox("Select a Paper", 
    ["Attention is All You Need", "BERT", "GPT-3", "Diffusion Models", "LLaMA"])
explanation_type = st.selectbox("Select Explanation Type", 
    ["Beginner Friendly", "Technical Summary", "Key Takeaways"])
length = st.selectbox("Select Output Length", 
    ["5 lines", "10 lines", "15 lines"])
```
- `st.selectbox()`: Creates dropdown menus for user choices
- These selections will become variables in our prompt template

**Step 2: Template Definition with Placeholders**
```python
template = """
Please provide a detailed explanation of the following paper:

Paper Name: {paper_name}
Explanation Type: {explanation_type}
Output Length: {length}
1. If mathematical equations are present, explain them in simple terms.
2. Provide a beginner-friendly overview of the paper.
3. Include a technical summary for advanced readers.
4. Use related examples to illustrate key concepts.
"""
```
- `{paper_name}`, `{explanation_type}`, `{length}`: Placeholders for variables
- Template contains both dynamic content and static instructions

**Step 3: PromptTemplate Creation and Usage**
```python
prompt_template = PromptTemplate(
    template=template,
    input_variables=["paper_name", "explanation_type", "length"]
)

prompt = prompt_template.invoke({
    "paper_name": paper_name,
    "explanation_type": explanation_type,
    "length": length
})
```
- `PromptTemplate`: Creates a reusable template object
- `input_variables`: Must match the placeholders in the template
- `invoke()`: Replaces placeholders with actual values

**Step 4: Model Invocation**
```python
if st.button("Generate Explanation"):
    response = model.invoke(prompt)
    st.write(response.content)
```
- The `prompt` object contains the fully formatted text with variables replaced
- Model receives a complete, customized prompt based on user selections

**Key concepts learned:**
- PromptTemplate class for reusable prompts
- Variable substitution with placeholders `{variable}`
- Dynamic prompt generation based on user input
- Structured prompt design with clear instructions

**Use cases:** Academic paper explanations, customized content generation

---

### 3. **Basic Chatbot** (`3.basic_chatbot.py`)

**What it does:**
- Creates a simple command-line chatbot
- Implements basic conversation loop with exit conditions
- Demonstrates direct model invocation without message structure

**Code breakdown:**

**Step 1: Model Initialization**
```python
from langchain_ollama import ChatOllama

model = ChatOllama(model="llama3.1", temperature=0.7)
```
- Sets up the LLM with the Llama 3.1 model
- `temperature=0.7`: Balanced creativity setting

**Step 2: Conversation Loop with Exit Handling**
```python
while True:
    user = input("You: ")
    if user.lower() in ["exit", "quit"]:
        print("Exiting the chat.")
        break
```
- `while True`: Creates an infinite conversation loop
- `input()`: Gets user input from command line
- Exit condition: Allows user to type "exit" or "quit" to end conversation

**Step 3: Model Invocation and Response**
```python
    llm = model.invoke(user)
    print("LLM:", llm.content)
```
- `model.invoke(user)`: Sends user input directly as a string (not as structured messages)
- `llm.content`: Extracts the text response from the model's output
- No conversation history is maintained - each exchange is independent

**Key concepts learned:**
- Basic conversation loop implementation
- Direct string input to model (no message structure)
- Exit condition handling
- Command-line interface design

**Limitations:** No conversation memory, no system prompts, simple text-in/text-out

---

### 4. **Template Store Use** (`4.template_store_use/`)

This section demonstrates template persistence and reusability.

#### **4a. Template Generation** (`template_gen.py`)

**What it does:**
- Creates a PromptTemplate and saves it to a JSON file
- Demonstrates template validation and persistence
- Enables template reuse across different applications

**Code breakdown:**

**Step 1: Template String Definition**
```python
template = """
Please provide a detailed explanation of the following paper:

Paper Name: {paper_name}
Explanation Type: {explanation_type}
Output Length: {length}
[Additional instructions...]
"""
```
- Multi-line string containing the template with placeholders
- Same template structure as in the previous example

**Step 2: Template Configuration**
```python
input_variables = ["paper_name", "explanation_type", "length"]
validate_template = True
```
- `input_variables`: List of placeholder names that must be provided
- `validate_template=True`: Ensures template syntax is correct before saving

**Step 3: Template Object Creation and Persistence**
```python
template_instance = PromptTemplate(
    template=template,
    input_variables=input_variables,
    validate_template=validate_template
)

template_instance.save("dynamic_prompt_template.json")
```
- Creates a PromptTemplate object with validation enabled
- `save()`: Serializes the template to a JSON file for later use
- The JSON file can now be shared across different applications

#### **4b. Template Loading** (`new_prompt_ui.py`)

**What it does:**
- Loads previously saved templates from JSON files
- Demonstrates template reuse in different applications
- Maintains consistency across multiple implementations

**Code breakdown:**

**Step 1: Template Loading**
```python
from langchain_core.prompts import load_prompt

prompt_template = load_prompt("dynamic_prompt_template.json")
```
- `load_prompt()`: Reads and reconstructs the PromptTemplate from JSON
- No need to redefine the template string or input variables

**Step 2: Template Usage (Same as Before)**
```python
prompt = prompt_template.invoke({
    "paper_name": paper_name,
    "explanation_type": explanation_type,
    "length": length
})
```
- The loaded template works identically to a manually created one
- Same `invoke()` method with the same variable dictionary

**Key concepts learned:**
- Template loading from JSON files
- Template versioning and management
- Separation of template definition and usage

---

### 5. **Basics of Messages** (`5.basics_of_messages.py`)

**What it does:**
- Demonstrates the three core message types in LangChain
- Shows how to build conversation structure programmatically
- Illustrates message chaining and conversation flow

**Code breakdown:**

**Step 1: Message List Creation**
```python
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

sample_messages = [
    HumanMessage(content="What is the capital of France?"),
    SystemMessage(content="The capital of France is Paris.")
]
```
- `HumanMessage`: Represents what a user would ask
- `SystemMessage`: Provides factual information or context
- List structure: Messages are processed in order

**Step 2: Model Invocation with Message List**
```python
model = ChatOllama(model="deepscaler", temperature=0.7)
result = model.invoke(sample_messages)
```
- Model receives the entire message list as context
- It can see both the question and the system-provided information

**Step 3: Adding AI Response to History**
```python
sample_messages.append(AIMessage(content=result.content))
print(sample_messages)
```
- `AIMessage`: Stores the AI's response for conversation history
- `append()`: Adds the response to the message list
- This creates a complete conversation record: [Human → System → AI]

**Final message list structure:**
```
[
    HumanMessage(content="What is the capital of France?"),
    SystemMessage(content="The capital of France is Paris."),
    AIMessage(content="The capital of France is indeed Paris...")
]
```

**Key concepts learned:**
- **HumanMessage**: Represents user input
- **SystemMessage**: Sets AI behavior and context
- **AIMessage**: Stores AI responses for conversation history
- Message list construction and management
- Conversation state building

**Message types explained:**
- **SystemMessage**: Instructions/role for the AI (not visible to user)
- **HumanMessage**: User's actual input/questions
- **AIMessage**: AI's responses (for maintaining conversation context)

---

### 6. **Build Memory in Chat** (`6.build_memory_in_chat/modified_bot.py`)

**What it does:**
- Implements conversation memory by maintaining message history
- Upgrades the basic chatbot with context awareness
- Demonstrates manual memory management

**Code breakdown:**

**Step 1: Initialize Conversation History**
```python
simple_history = [
    SystemMessage(content="You are a helpful assistant.")
]
```
- Starts with a system message to establish AI behavior
- This message persists throughout the entire conversation
- `simple_history`: List that will grow with each exchange

**Step 2: User Input and History Updates**
```python
while True:
    user = input("You: ")
    simple_history.append(HumanMessage(content=user))
    
    if user.lower() in ["exit", "quit"]:
        break
```
- Gets user input and immediately adds it to history
- Each user message becomes part of the permanent conversation record
- Exit condition allows graceful termination

**Step 3: Model Invocation with Full History**
```python
    result = model.invoke(simple_history)
    simple_history.append(AIMessage(content=result.content))
    print("LLM:", result.content)
```
- `model.invoke(simple_history)`: Model sees the entire conversation
- AI response is added to history for future context
- Each turn, the history grows: [System, Human₁, AI₁, Human₂, AI₂, ...]

**Memory evolution example:**
```
Turn 1: [System, Human: "Hello"]
        → [System, Human: "Hello", AI: "Hi there!"]

Turn 2: [System, Human: "Hello", AI: "Hi there!", Human: "What's your name?"]
        → [System, Human: "Hello", AI: "Hi there!", Human: "What's your name?", AI: "I'm an assistant..."]
```

**Key concepts learned:**
- Manual conversation memory implementation
- Message history accumulation
- Context preservation across turns
- System message as conversation foundation

**Advantages over basic chatbot:**
- Remembers previous exchanges
- Maintains conversation context
- Can reference earlier parts of conversation

---

### 7. **Dynamic Message ChatPrompt** (`7.dynamic_message_chatprompt.py`)

**What it does:**
- Introduces ChatPromptTemplate for structured conversation templates
- Demonstrates variable substitution in message templates
- Shows simplified message creation syntax

**Code breakdown:**

**Step 1: ChatPromptTemplate Definition**
```python
from langchain_core.prompts import ChatPromptTemplate

chat_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant in the field of {domain}."),
    ("human", "Explain the concept of {concept} in simple terms."),
])
```
- `from_messages()`: Creates a template from a list of message tuples
- `("system", "content")`: Simplified syntax instead of `SystemMessage(content="content")`
- `{domain}` and `{concept}`: Placeholders for variable substitution

**Step 2: Template Invocation with Variables**
```python
result = chat_template.invoke({
    "domain": "science",
    "concept": "photosynthesis"
})
```
- `invoke()`: Replaces placeholders with actual values
- Returns a formatted message list ready for model consumption

**Step 3: What the Template Produces**
The `result` contains:
```python
[
    SystemMessage(content="You are a helpful assistant in the field of science."),
    HumanMessage(content="Explain the concept of photosynthesis in simple terms.")
]
```
- Template automatically creates proper message objects
- Variables are substituted in both system and human messages
- Ready to send directly to a chat model

**Advantages over manual message creation:**
```python
# Manual way (more verbose):
messages = [
    SystemMessage(content=f"You are a helpful assistant in the field of {domain}."),
    HumanMessage(content=f"Explain the concept of {concept} in simple terms.")
]

# Template way (cleaner):
result = chat_template.invoke({"domain": domain, "concept": concept})
```

**Key concepts learned:**
- ChatPromptTemplate for structured conversations
- Simplified message syntax: `("system", "content")` vs `SystemMessage(content="content")`
- Variable substitution in both system and human messages
- Template-based conversation initialization

**Advantages:**
- More readable than manual message creation
- Built-in variable substitution
- Structured conversation patterns

---

### 8. **Dynamic Chat History** (`8.dynamic_chat_history/`)

This demonstrates the most advanced conversation memory management.

#### **Chat History Storage** (`chat_history.txt`)

**What it contains:**
```
HumanMessage(content="When will i get my refund?"),
SystemMessage(content="You will get you refund within 2 business days.")
```

**Purpose:** Stores conversation history that can be loaded and used in new conversations

#### **Dynamic History Loading** (`dynamic_chat_history.py`)

**What it does:**
- Uses MessagesPlaceholder for dynamic conversation injection
- Loads conversation history from external files
- Demonstrates advanced template composition

**Code breakdown:**

**Step 1: Template with MessagesPlaceholder**
```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

chat_prompt = ChatPromptTemplate([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{user_input}")
])
```
- `MessagesPlaceholder`: A special placeholder for a list of messages
- `variable_name="chat_history"`: The key to use when invoking the template
- This placeholder can accept any number of messages dynamically

**Step 2: Loading History from File**
```python
chat_history = []
with open("chat_history.txt", "r") as file:
    chat_history.extend(file.readlines())
```
- Reads conversation history from an external file
- `extend()`: Adds all file lines to the chat_history list
- File contains: `HumanMessage(content="When will i get my refund?"), SystemMessage(content="You will get you refund within 2 business days.")`

**Step 3: Template Invocation with Dynamic History**
```python
result = chat_prompt.invoke({
    "chat_history": chat_history,
    "user_input": "When will I get my refund?"
})
```
- `chat_history`: List of previous messages (loaded from file)
- `user_input`: Current user question
- Template dynamically inserts the history between system and human messages

**Final Template Structure:**
```
[
    SystemMessage(content="You are a helpful assistant."),
    # chat_history messages inserted here dynamically:
    HumanMessage(content="When will i get my refund?"),
    SystemMessage(content="You will get you refund within 2 business days."),
    # Current user input:
    HumanMessage(content="When will I get my refund?")
]
```

**Key insight:** `MessagesPlaceholder` allows templates to work with variable-length conversation histories, making them suitable for long-running conversations or customer service scenarios where previous interactions matter.

**Key concepts learned:**
- **MessagesPlaceholder**: Dynamic insertion of conversation history
- External conversation storage and retrieval
- Template composition with variable message counts
- Advanced conversation context management

**Use cases:**
- Customer service bots that remember previous interactions
- Multi-session conversations
- Conversation analytics and review

---

## 🎯 Learning Path Summary

### **Beginner Level** (Files 1-3)
- Start with `1.basic_prompt_ui.py` for UI basics
- Learn dynamic prompts with `2.dynamic_prompt_basic.py`
- Understand basic loops with `3.basic_chatbot.py`

### **Intermediate Level** (Files 4-5)
- Master template management with `4.template_store_use/`
- Learn message types with `5.basics_of_messages.py`

### **Advanced Level** (Files 6-8)
- Implement memory with `6.build_memory_in_chat/`
- Use advanced templates with `7.dynamic_message_chatprompt.py`
- Master dynamic history with `8.dynamic_chat_history/`

## 🛠️ Key Technologies & Concepts

### **Core LangChain Components:**
- `ChatOllama`: Local LLM integration
- `PromptTemplate`: Dynamic prompt generation
- `ChatPromptTemplate`: Structured conversation templates
- `MessagesPlaceholder`: Dynamic message injection

### **Message Types:**
- `SystemMessage`: AI behavior/role definition
- `HumanMessage`: User input representation
- `AIMessage`: AI response storage

### **Advanced Features:**
- Template persistence (JSON save/load)
- Conversation memory management
- Dynamic content generation
- Multi-turn conversation handling

### **UI Technologies:**
- Streamlit for web interfaces
- Command-line for simple interactions
- File-based conversation storage

## 🚀 Running the Examples

### **Streamlit Applications:**
```bash
streamlit run 1.basic_prompt_ui.py
streamlit run 2.dynamic_prompt_basic.py
streamlit run 4.template_store_use/new_prompt_ui.py
```

### **Command-Line Applications:**
```bash
python 3.basic_chatbot.py
python 5.basics_of_messages.py
python 6.build_memory_in_chat/modified_bot.py
python 7.dynamic_message_chatprompt.py
python 8.dynamic_chat_history/dynamic_chat_history.py
```

### **Template Management:**
```bash
# First create the template
python 4.template_store_use/template_gen.py
# Then use it
streamlit run 4.template_store_use/new_prompt_ui.py
```

## 💡 Best Practices Learned

1. **Always use SystemMessage** to define AI behavior
2. **Maintain conversation history** for context awareness
3. **Use templates** for reusable prompt patterns
4. **Validate templates** before saving
5. **Store conversations** for analysis and continuity
6. **Structure messages properly** for best results
7. **Handle exit conditions** in conversation loops

## 🎯 Next Steps

After completing this tutorial, you'll be ready for:
- Building production chatbots with persistent memory
- Creating complex conversational AI applications
- Implementing custom prompt engineering patterns
- Developing multi-user conversation systems
- Integrating with vector databases for RAG applications

## 🔧 Dependencies

```bash
pip install langchain-ollama langchain-core streamlit
```

**Models Used:** phi3, llama3.1, deepscaler, gemma:2b (via Ollama)
