# LangChain Models Collection

This repository contains examples and implementations of various LangChain model integrations, including Large Language Models (LLMs), Chat Models, and Embedding Models using different providers and deployment methods.

## 📁 Project Structure

```terminal
langchain_models/
├── LLMs/                    # Large Language Models
│   └── llm_demo.py         # Ollama LLM integration example
├── ChatModels/             # Conversational AI Models
│   ├── chatmodel_hf_using_api.py    # HuggingFace API chat model
│   ├── chatmodel_hf_using_local.py  # Local HuggingFace chat model
│   ├── chatmodel_ollama.py          # Ollama chat model
│   └── Readme.md                    # ChatModels documentation
├── EmbeddingModels/        # Text Embedding Models
│   ├── embedding_hf_local.py       # Local HuggingFace embeddings
│   ├── document_similarity.py      # Document similarity analysis
│   └── Readme.md                   # EmbeddingModels documentation
└── Readme.md              # This file
```

## 🚀 Quick Start

### Prerequisites

```bash
pip install langchain langchain-huggingface langchain-ollama
pip install transformers sentence-transformers
pip install scikit-learn numpy python-dotenv
```

### Environment Setup

Create a `.env` file in the root directory:

```env
HUGGINGFACEHUB_API_TOKEN="your-hugging-face-token-here"
OPENAI_API_KEY="your-openai-api-key-here"
```

## 📚 Model Categories

### 1. Large Language Models (LLMs)

**Location**: `LLMs/`

Basic text generation models for completion tasks.

#### **llm_demo.py** - Ollama Integration

```python
# Simple LLM for text completion
from langchain_ollama import OllamaLLM

llm = OllamaLLM(model="phi3", temperature=0.7)
response = llm.invoke("What is the capital of France?")
```

**What this code does:**
- Creates a connection to the locally running Ollama service
- Initializes the Phi3 model with moderate creativity (temperature=0.7)
- Sends a simple text prompt and receives a completion response
- Returns a string answer without conversation context

**Features:**

- Local Ollama model integration
- Simple question-answering
- No conversation memory
- Fast inference with Phi3 model

---

### 2. Chat Models

**Location**: `ChatModels/`

Conversational AI models with message-based interactions.

#### **chatmodel_ollama.py** - Local Ollama Chat

```python
# Conversational AI with system prompts
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

chat = ChatOllama(model="phi3", temperature=0.7)
messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="Hello!")
]
response = chat.invoke(messages)
```

**What this code does:**
- Creates a chat-based model that can handle structured conversations
- Sets up a system message to define the AI's role and behavior
- Sends a list of messages (system + human) to maintain conversation context
- Returns a structured chat response that can be part of ongoing dialogue
- Enables more sophisticated interactions than simple text completion

**Features:**

- Message-based conversation
- System prompt configuration
- Local model deployment
- Conversation context handling

#### **chatmodel_hf_using_local.py** - Local HuggingFace Chat

```python
# Local HuggingFace model with custom parameters
from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline

generator = pipeline(
    model="./models/falcon-1b",
    task="text-generation",
    max_length=100,
    temperature=0.5
)
llm = HuggingFacePipeline(pipeline=generator)
```

**What this code does:**
- Downloads and loads a HuggingFace model locally to your machine
- Creates a text generation pipeline with custom parameters (length, temperature)
- Wraps the HuggingFace pipeline in a LangChain-compatible interface
- Enables offline text generation without requiring internet connectivity
- Provides full control over model parameters and generation settings

**Features:**

- Local HuggingFace model integration
- Custom generation parameters
- Offline functionality
- Privacy-focused deployment

#### **chatmodel_hf_using_api.py** - HuggingFace API Chat

```python
# HuggingFace Inference API integration
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1",
    task="text-generation"
)
chat_model = ChatHuggingFace(llm=llm)
```

**What this code does:**
- Connects to HuggingFace's cloud-based Inference API service
- Accesses the DeepSeek-R1 model remotely without local installation
- Creates an API endpoint connection that handles authentication automatically
- Wraps the API endpoint in a chat-compatible interface for conversations
- Enables access to large, powerful models that may be too big to run locally

**Features:**

- Cloud-based inference
- Access to latest models
- Scalable deployment
- API-based integration

---

### 3. Embedding Models

**Location**: `EmbeddingModels/`

Text embedding models for semantic understanding and similarity analysis.

#### **embedding_hf_local.py** - Local Embeddings

```python
# Local sentence transformer embeddings
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="./models/sentence-transformers/all-MiniLM-L6-v2"
)
result = embeddings.embed_query("What is the capital of Bangladesh?")
```

**What this code does:**
- Loads a pre-trained sentence transformer model locally for text embeddings
- Converts text into numerical vectors (384 dimensions) that capture semantic meaning
- Takes any text query and transforms it into a dense vector representation
- Enables semantic similarity comparisons between different pieces of text
- Provides the foundation for semantic search, clustering, and similarity analysis

**Features:**

- 384-dimensional embeddings
- Local model deployment
- Fast inference
- Semantic similarity analysis

#### **document_similarity.py** - Similarity Analysis

```python
# Document similarity comparison
from sklearn.metrics.pairwise import cosine_similarity

documents = [
    "The capital of Bangladesh is Dhaka.",
    "The capital of France is Paris.",
    "Dhaka is the largest city in Bangladesh."
]

embeddings_list = embeddings.embed_documents(documents)
similarity_matrix = cosine_similarity(embeddings_list)
```

**What this code does:**
- Takes multiple text documents and converts each into embedding vectors
- Uses cosine similarity to measure how semantically similar documents are to each other
- Creates a similarity matrix showing the relationship between all document pairs
- Values range from 0 (completely different) to 1 (identical meaning)
- Identifies which documents discuss similar topics or contain related information

**Features:**

- Multi-document comparison
- Cosine similarity calculation
- Similarity matrix generation
- Content recommendation capabilities

## 🛠️ Use Cases

### 1. **Question Answering**

- **Files**: `llm_demo.py`, `chatmodel_ollama.py`
- **Description**: Simple Q&A systems using local models
- **Best for**: Quick answers, factual queries

### 2. **Conversational AI**

- **Files**: All chat model files
- **Description**: Interactive chatbots with conversation memory
- **Best for**: Customer support, virtual assistants

### 3. **Content Generation**

- **Files**: `chatmodel_hf_using_local.py`
- **Description**: Generate articles, summaries, creative content
- **Best for**: Content creation, writing assistance

### 4. **Semantic Search**

- **Files**: `embedding_hf_local.py`, `document_similarity.py`
- **Description**: Find similar documents based on meaning
- **Best for**: Document retrieval, recommendation systems

### 5. **Document Analysis**

- **Files**: `document_similarity.py`
- **Description**: Analyze document relationships and clusters
- **Best for**: Content organization, duplicate detection

## 🔧 Model Specifications

| Model Type | Provider | Model Name | Size | Use Case |
|------------|----------|------------|------|----------|
| LLM | Ollama | Phi3 | ~2.3GB | General text completion |
| Chat | Ollama | Phi3 | ~2.3GB | Conversational AI |
| Chat | HuggingFace | Falcon-1B | ~2GB | Local chat generation |
| Chat | HuggingFace API | DeepSeek-R1 | Cloud | Advanced reasoning |
| Embedding | HuggingFace | all-MiniLM-L6-v2 | ~90MB | Semantic embeddings |

## 🚀 Performance Tips

### Local Models

- **GPU Acceleration**: Use CUDA if available
- **Memory Management**: Monitor RAM usage with large models
- **Model Quantization**: Use half-precision for memory efficiency

### API Models

- **Rate Limiting**: Respect API rate limits
- **Caching**: Cache responses for repeated queries
- **Error Handling**: Implement retry mechanisms

### Embeddings

- **Batch Processing**: Process multiple documents together
- **Vector Storage**: Use vector databases for large datasets
- **Similarity Thresholds**: Tune similarity thresholds for your use case

## 📖 Getting Started Guide

### 1. **Start with LLMs** (`LLMs/llm_demo.py`)

- Simple text completion
- No complex setup required
- Good for understanding basics

### 2. **Move to Chat Models** (`ChatModels/`)

- More sophisticated interactions
- Conversation context
- System prompts

### 3. **Explore Embeddings** (`EmbeddingModels/`)

- Semantic understanding
- Document similarity
- Search applications

## 🔍 Advanced Features

### Chain Integration

All models can be used with LangChain chains:

```python
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

chain = LLMChain(llm=your_model, prompt=your_prompt)
```

### Memory Integration

Add conversation memory:

```python
from langchain.memory import ConversationBufferMemory
memory = ConversationBufferMemory()
```

### Agent Framework

Use models with LangChain agents:

```python
from langchain.agents import initialize_agent
agent = initialize_agent(tools, llm, agent_type="...")
```

## 🛡️ Best Practices

1. **Security**: Never commit API keys to version control
2. **Error Handling**: Implement try-catch blocks for production
3. **Monitoring**: Track model performance and costs
4. **Testing**: Test with various inputs and edge cases
5. **Documentation**: Keep README files updated

## 🔗 Related Resources

- [LangChain Documentation](https://python.langchain.com/)
- [HuggingFace Models](https://huggingface.co/models)
- [Ollama Models](https://ollama.ai/library)
- [Sentence Transformers](https://www.sbert.net/)

## 📝 Contributing

When adding new models or examples:

1. Follow the existing folder structure
2. Include comprehensive documentation
3. Add usage examples
4. Update this main README
5. Test thoroughly before committing

## 🆘 Troubleshooting

### Common Issues

- **Model not found**: Check local model paths
- **API errors**: Verify API keys and internet connection
- **Memory errors**: Reduce batch sizes or use CPU
- **Slow performance**: Consider model quantization or GPU usage

For detailed troubleshooting, check the README files in each subfolder.
