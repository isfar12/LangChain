# 🦜⛓️ LangChain Complete Learning Repository

![LangChain](https://img.shields.io/badge/LangChain-Latest-blue.svg)
![Ollama](https://img.shields.io/badge/Ollama-Local_LLMs-green.svg)
![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Status](https://img.shields.io/badge/Status-Complete-success.svg)

A comprehensive, hands-on learning repository covering all aspects of LangChain from basics to advanced AI agent development. This repository contains tutorials, projects, and practical implementations using local Ollama models.

---

## 📊 **Available Ollama Models**

### **🤖 Chat Models (Tool Calling Supported)**
| Model Name | ID | Size | Tool Calling | Use Case |
|------------|----|----|-------------|----------|
| `mistral:7b` | 6577803aa9a0 | 4.4 GB | ✅ **Yes** | General purpose, tool calling |
| `llama3.2:latest` | a80c4f17acd5 | 2.0 GB | ✅ **Yes** | Conversation, reasoning |
| `llama3-groq-tool-use:latest` | 36211dad2b15 | 4.7 GB | ✅ **Yes** | Specialized for tool usage |
| `deepseek-r1:8b` | 6995872bfe4c | 5.2 GB | ✅ **Yes** | Advanced reasoning |
| `qwen3:4b` | 2bfd38a7daaf | 2.6 GB | ✅ **Yes** | Multilingual support |

### **🚫 Models WITHOUT Tool Calling Support**
| Model Name | ID | Size | Limitation | Best For |
|------------|----|----|-----------|----------|
| `gemma3:4b` | a2af6cc3eb7f | 3.3 GB | ❌ **No Tool Calling** | Text generation only |
| `gemma2:2b` | 8ccf136fdd52 | 1.6 GB | ❌ **No Tool Calling** | Lightweight chat |
| `gemma3:1b` | 8648f39daa8f | 815 MB | ❌ **No Tool Calling** | Resource-constrained environments |

### **📊 Embedding Models**
| Model Name | ID | Size | Purpose |
|------------|----|----|---------|
| `mxbai-embed-large:latest` | 468836162de7 | 669 MB | High-quality embeddings |
| `nomic-embed-text:latest` | 0a109f422b47 | 274 MB | Lightweight embeddings |

> **⚠️ Important Note**: Gemma models do not support tool calling functionality. Use Mistral, Llama, or Qwen models for tool-based projects.

---

## 🗂️ **Repository Structure & Learning Path**

### **🌟 Core Learning Modules**

#### **1. 📚 Foundation Knowledge**
```
1. introduction.md              # LangChain concepts and workflow
langchain_notes_by_campusx.pdf # Comprehensive reference guide
langchain_packages.txt         # Required Python packages
```

#### **2. 🧠 Models & Embeddings** 
```
1. langchain_models/
├── 1.LLMs/                    # Large Language Models
├── 2.ChatModels/              # Conversational AI models  
├── 3.EmbeddingModels/         # Text embedding models
└── README.md                  # Model comparison and setup
```
**Learn**: Model types, initialization, configuration, and selection strategies.

#### **3. 💬 Prompt Engineering**
```
2. langchain_prompts/
├── 1.basic_prompt_ui.py              # Simple prompt templates
├── 2.dynamic_prompt_basic.py         # Dynamic prompt generation
├── 3.basic_chatbot.py               # Basic conversational AI
├── 4.template_store_use/            # Template management
├── 5.basics_of_messages.py          # Message handling
├── 6.build_memory_in_chat/          # Conversation memory
├── 7.dynamic_message_chatprompt.py  # Advanced messaging
├── 8.dynamic_chat_history/          # Chat history management
└── README.md                        # Comprehensive prompt guide
```
**Learn**: Prompt templates, dynamic content, conversation management, and memory systems.

#### **4. 📋 Structured Output**
```
3. structured_output/
├── 1.typedict/                      # Basic type hints
├── 2.pydantic/                      # Advanced validation ⭐ RECOMMENDED
├── 3.json/                          # JSON Schema approach
└── Readme.md                        # Complete tutorial guide
```
**Learn**: Getting structured, predictable responses from LLMs using three different approaches.

#### **5. 🔧 Output Parsers**
```
4. output_parsers/
├── 1. without_stroutputparser.py    # Raw output handling
├── 2. stroutputparser.py           # String parsing
├── 3. jsonoutputparser.py          # JSON parsing  
├── 4. structured_output_using_parsers.py  # Advanced parsing
├── 5. structured_output_using_pydantic.py # Pydantic integration
└── Tutorial-*.md                   # Step-by-step guides
```
**Learn**: Processing and structuring LLM responses with various parsing techniques.

#### **6. ⛓️ Chains & Workflows**
```
5. chains_in_langchain/
├── 1. simple_chain.py              # Basic chaining
├── 2. parallel_chain.py            # Parallel execution
├── 3. conditional_chains.py        # Conditional logic
└── README.md                       # Chain orchestration guide
```
**Learn**: Combining multiple operations, parallel processing, and workflow management.

#### **7. 🔄 Runnables & Advanced Patterns**
```
6. runnables/
├── 1. runnable_sequence.py         # Sequential operations
├── 2. runnable_parellal.py         # Parallel processing
├── 3. runnable_passthrough.py      # Data passing
├── 4. runnable_lambda.py           # Custom functions
├── 5. runnable_branch.py           # Conditional branching
└── README.md                       # Advanced patterns guide
```
**Learn**: LangChain's newest execution framework for building complex applications.

### **🔍 Data Processing & Retrieval**

#### **8. 📄 Document Loading**
```
7. document_loaders/
├── 1. text_loader.py               # Plain text files
├── 2. pypdf_loader.py              # PDF documents
├── 3. directory_loader.py          # Batch loading
├── 4. lazy_loader_Vs_loader.py     # Performance optimization
├── 5. csv_loader.py                # CSV data
├── 5. web_based_loader.py          # Web scraping
├── Documents/                      # Sample documents
└── README.md                       # Loading strategies guide
```
**Learn**: Loading data from various sources for AI processing.

#### **9. ✂️ Text Processing**
```
8. text_splitters/
├── 1. length_based_split.py        # Character/token splitting
├── 2. text_structure_based.py      # Semantic splitting
├── 3. code_split.py                # Programming language aware
├── 4. semantic_bases.py            # Meaning-based chunks
└── README.md                       # Text chunking strategies
```
**Learn**: Breaking down large texts into manageable chunks for processing.

#### **10. 🗄️ Vector Storage**
```
9. vector_stores/
└── (Vector database implementations)
faiss_vector_store/                 # FAISS index storage
├── index.faiss                    # Vector index file
└── (Associated metadata)
```
**Learn**: Storing and retrieving embeddings for semantic search.

#### **11. 🔍 Retrievers**
```
10. retrievers/
├── 1. wikipedia_retriever.py       # Wikipedia integration
├── 2. vector_store_retriever.py    # Vector-based search
├── 3. mmr_retriever.py             # Maximum Marginal Relevance
├── 4. multi_query_retriever.py     # Multi-query expansion
├── 5. context_compress.py          # Context compression
├── langchain_retrievers.ipynb      # Interactive examples
└── README.md                       # Retrieval strategies guide
```
**Learn**: Different strategies for finding and retrieving relevant information.

### **🛠️ Advanced Applications**

#### **12. 🔧 Tools & Agents**
```
12. tools/
├── 1. using_tools_decorator.py     # Basic tool creation
├── 2. using_pydantic.py            # Structured tools
├── 3. basic_tool_calling.ipynb     # Interactive tool usage
├── 4. Project_Currency_convert.ipynb # Complete AI agent
└── README.md                       # Comprehensive tools guide
```
**Learn**: Creating AI agents that can interact with external systems and APIs.

#### **13. 🎥 Complete Projects**
```
11. Project_Youtube_chatbot/
├── transcript_extract.py          # YouTube transcript extraction
├── step_by_step.ipynb            # Development process
├── youtube_bot.py                # Basic implementation
├── structured_youtube_bot.py     # Advanced version
├── streamlit_youtube_bot_ui.py   # Interactive UI
└── README.md                     # Project documentation
```
**Learn**: End-to-end project development with real-world applications.

### **⚙️ Development Tools**
```
Local_LLMs_downloader/             # Model management utilities
├── downloader.py                 # Automated model setup
models/                           # Local model storage
```

---

## 🎯 **Recommended Learning Path**

### **🥇 Beginner (Week 1-2)**
1. **Start Here**: `1. introduction.md` - Understand LangChain concepts
2. **Models**: Explore `1. langchain_models/` - Learn model types
3. **Basic Prompts**: Try `2. langchain_prompts/` - Master prompt engineering
4. **Simple Output**: Begin with `3. structured_output/1.typedict/`

### **🥈 Intermediate (Week 3-4)**  
1. **Advanced Output**: Master `3. structured_output/2.pydantic/` ⭐
2. **Parsing**: Work through `4. output_parsers/` tutorials
3. **Chains**: Build workflows with `5. chains_in_langchain/`
4. **Document Processing**: Learn `7. document_loaders/` and `8. text_splitters/`

### **🥉 Advanced (Week 5-6)**
1. **Runnables**: Master `6. runnables/` for advanced patterns
2. **Vector Storage**: Implement `9. vector_stores/` and `10. retrievers/`
3. **Tools & Agents**: Build agents with `12. tools/` 
4. **Complete Projects**: Implement `11. Project_Youtube_chatbot/`

---

## 🚀 **Quick Start Guide**

### **Prerequisites**
```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull recommended models
ollama pull mistral:7b
ollama pull llama3.2:latest  
ollama pull mxbai-embed-large

# Install Python packages
pip install -r langchain_packages.txt
```

### **Test Your Setup**
```python
from langchain_ollama import ChatOllama

# Test chat model
llm = ChatOllama(model="mistral:7b")
response = llm.invoke("Hello! Can you help me learn LangChain?")
print(response.content)

# Test embedding model  
from langchain_ollama import OllamaEmbeddings
embeddings = OllamaEmbeddings(model="mxbai-embed-large")
vectors = embeddings.embed_query("Test embedding")
print(f"Embedding dimensions: {len(vectors)}")
```

---

## 📋 **Key Technologies & Dependencies**

### **Core LangChain Packages**
- `langchain` - Main framework
- `langchain-core` - Core components
- `langchain_ollama` - Ollama integration
- `langchain-experimental` - Cutting-edge features

### **Model Integrations**
- `langchain-openai` - OpenAI models
- `langchain_gemini` - Google Gemini
- `langchain-anthropic` - Claude models
- `langchain-huggingface` - HuggingFace models

### **Data Processing**
- `unstructured[all-docs]` - Document processing
- `langchain-chroma` - Vector database
- `pytesseract` - OCR capabilities
- `pdf2image` - PDF processing

### **Development Tools**
- `python-dotenv` - Environment management
- `numpy`, `scikit-learn` - Data science
- `transformers` - Model utilities

---

## 🎯 **Project Highlights**

### **🏆 Featured Projects**

#### **AI Currency Converter Agent** (`12. tools/4. Project_Currency_convert.ipynb`)
- **Real-time API integration** with exchange rate services
- **Tool coordination** preventing AI hallucination
- **InjectedToolArg** for parameter control
- **Production-ready** error handling

#### **YouTube Chatbot** (`11. Project_Youtube_chatbot/`)
- **Video transcript extraction** and processing
- **RAG implementation** for Q&A systems
- **Streamlit UI** for interactive experience
- **Structured conversation** management

#### **Vector Search System** (`9. vector_stores/` + `10. retrievers/`)
- **FAISS vector database** implementation
- **Multiple retrieval strategies** (MMR, multi-query, compression)
- **Semantic search** capabilities
- **Performance optimization** techniques

---

## 💡 **Best Practices Learned**

### **🔧 Model Selection**
- **Tool Calling**: Use Mistral, Llama3.2, or Qwen (avoid Gemma models)
- **Embeddings**: mxbai-embed-large for quality, nomic-embed-text for speed
- **Temperature**: 0.0 for structured output, 0.7 for creative tasks

### **📊 Structured Output**
- **Pydantic recommended** for most use cases
- **TypedDict** for simple type hints
- **JSON Schema** for complex validation rules

### **⛓️ Chain Design**
- **Keep chains focused** on single responsibilities
- **Use parallel processing** when possible
- **Implement error handling** at each step

### **🗄️ Vector Storage**
- **Chunk size**: 1000-1500 characters for most documents
- **Overlap**: 200-300 characters between chunks
- **Metadata filtering** for improved retrieval

---

## 🔧 **Troubleshooting Common Issues**

### **Model-Related**
- **Tool calling not working**: Ensure you're using supported models (not Gemma)
- **Out of memory**: Use smaller models (llama3.2 instead of larger variants)
- **Slow responses**: Consider model size vs. performance trade-offs

### **Development Issues**  
- **Import errors**: Check `langchain_packages.txt` for required packages
- **API timeouts**: Implement retry logic and error handling
- **Token limits**: Use text splitters for large documents

### **Performance Optimization**
- **Caching**: Implement LRU cache for frequently accessed data
- **Batch processing**: Process multiple items together
- **Lazy loading**: Load resources only when needed

---

## 🎯 **Next Steps & Advanced Topics**

After completing this repository, you'll be ready for:

### **🚀 Production Applications**
- **Scalable RAG systems** with enterprise data
- **Multi-agent coordination** for complex workflows  
- **API service development** with FastAPI/Flask
- **Real-time processing** with streaming responses

### **🧠 Advanced AI Patterns**
- **Multi-modal applications** (text, image, audio)
- **Fine-tuning local models** for specific domains
- **Custom tool development** for specialized tasks
- **Integration with external services** (databases, APIs, IoT)

### **📊 Enterprise Features**
- **Monitoring and observability** with LangSmith
- **Security and compliance** implementations
- **Cost optimization** strategies
- **A/B testing** for AI applications

---

## 🤝 **Contributing**

This repository represents a complete learning journey. Feel free to:
- **Fork** and customize for your learning needs
- **Submit issues** for clarifications or improvements
- **Share** your own implementations and variations
- **Star** the repository if you find it helpful

---

## 📚 **Additional Resources**

- **LangChain Documentation**: https://python.langchain.com/
- **Ollama Models**: https://ollama.ai/library
- **LangSmith for Monitoring**: https://smith.langchain.com/
- **Community Discord**: https://discord.gg/langchain

---

**⭐ Star this repository if it helped you master LangChain!**

*Last Updated: August 10, 2025*
