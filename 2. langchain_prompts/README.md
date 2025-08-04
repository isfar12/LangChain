# LangChain Prompts

This folder contains examples demonstrating various prompt engineering techniques and message handling in LangChain.

## What's Covered

### Basic Prompting

- **1.basic_prompt_ui.py**: Simple Streamlit interface for chat interactions using system and human messages
- **3.basic_chatbot.py**: Command-line chatbot with basic message handling

### Dynamic Prompts

- **2.dynamic_prompt_basic.py**: Dynamic prompt generation using PromptTemplate with customizable options (paper selection, explanation type, output length)
- **7.dynamic_message_chatprompt.py**: ChatPromptTemplate for creating structured conversations with variable substitution

### Message Types

- **5.basics_of_messages.py**: Working with different message types (HumanMessage, SystemMessage, AIMessage)

### Template Management

- **4.template_store_use/**:
  - Save and load prompt templates as JSON files
  - Template validation and reusability

### Chat Memory

- **6.build_memory_in_chat/**: Implementing conversation memory by maintaining message history
- **8.dynamic_chat_history/**: Using MessagesPlaceholder for dynamic chat history injection

## Key Concepts Learned

1. **Message Structure**: Understanding HumanMessage, SystemMessage, and AIMessage classes
2. **Prompt Templates**: Creating reusable templates with variable substitution using `PromptTemplate`
3. **Chat Templates**: Using `ChatPromptTemplate` for structured conversation flows
4. **Memory Management**: Maintaining conversation context through message history
5. **Dynamic Prompting**: Building flexible prompts based on user input and selections
6. **Template Persistence**: Saving and loading prompt templates for reuse

## Models Used

- Ollama models: phi3, llama3.1, deepscaler, gemma:2b

## Dependencies

- langchain_ollama
- langchain_core
- streamlit (for UI examples)
