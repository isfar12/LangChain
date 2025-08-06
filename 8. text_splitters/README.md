# Text Splitters in LangChain

## Table of Contents

1. [Introduction to Text Splitters](#introduction-to-text-splitters)
2. [Why Text Splitting is Important](#why-text-splitting-is-important)
3. [Prerequisites](#prerequisites)
4. [Tutorial 1: Length-Based Text Splitting](#tutorial-1-length-based-text-splitting)
5. [Tutorial 2: Structure-Based Text Splitting](#tutorial-2-structure-based-text-splitting)
6. [Tutorial 3: Code-Specific Text Splitting](#tutorial-3-code-specific-text-splitting)
7. [Tutorial 4: Semantic-Based Text Splitting](#tutorial-4-semantic-based-text-splitting)
8. [Comparison and Use Cases](#comparison-and-use-cases)
9. [Best Practices](#best-practices)
10. [Running the Examples](#running-the-examples)

## Introduction to Text Splitters

Text splitters are essential tools in LangChain that break down large documents into smaller, manageable chunks. These chunks are crucial for:

- **Vector databases**: Most embedding models have token limits
- **RAG (Retrieval Augmented Generation)**: Smaller chunks improve retrieval accuracy
- **Memory management**: Processing large documents efficiently
- **Context windows**: Fitting content within model limits

### Core Concept

When you have a 10,000-word document, you can't send it directly to most language models due to context window limitations. Text splitters solve this by:

1. **Breaking** large text into smaller pieces
2. **Preserving** context through overlapping chunks
3. **Maintaining** semantic meaning when possible
4. **Optimizing** for downstream tasks like embedding and retrieval

## Why Text Splitting is Important

### Problem Without Text Splitting
- Language models have token limits (e.g., 4096, 8192 tokens)
- Large documents exceed these limits
- Information gets truncated or lost
- Poor retrieval performance in RAG systems

### Solution With Text Splitting
- Documents fit within model constraints
- Better granularity for retrieval
- Improved accuracy in question-answering
- Efficient processing of large datasets

## Prerequisites

Before running these examples, ensure you have:

1. **Required Python packages**:
   ```bash
   pip install langchain langchain-community langchain-ollama langchain-huggingface langchain-experimental
   ```

2. **For semantic splitting**:
   ```bash
   pip install sentence-transformers
   ollama pull mxbai-embed-large
   ```

3. **Sample documents**: The examples reference PDF files from the document_loaders folder

## Tutorial 1: Length-Based Text Splitting

**File**: `1. length_based_split.py`

### Concept

`CharacterTextSplitter` is the simplest text splitter that divides text based on character count. It splits text at specified separators and creates chunks of a fixed size.

### Key Parameters

- **chunk_size**: Maximum characters per chunk
- **chunk_overlap**: Characters to overlap between chunks
- **separator**: Character(s) to split on (empty string means any character)

### Code Explanation

```python
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

splitter = CharacterTextSplitter(
    chunk_size=100,        # Each chunk max 100 characters
    chunk_overlap=0,       # No overlap between chunks
    separator=''           # Split at any character boundary
)
```

**Setup Parameters**:
- `chunk_size=100`: Very small chunks for demonstration
- `chunk_overlap=0`: Clean separation between chunks
- `separator=''`: Splits anywhere, not respecting word boundaries

### Two Splitting Methods

**1. Text Splitting (`split_text`)**:
```python
text = 'Document loaders in LangChain are responsible for...'
result = splitter.split_text(text)
```
- Input: Plain string
- Output: List of text chunks
- Use case: Simple text processing

**2. Document Splitting (`split_documents`)**:
```python
pdf_loader = PyPDFLoader(file_path=r"...")
docs = pdf_loader.load()
result_docs = splitter.split_documents(docs)
```
- Input: List of Document objects
- Output: List of Document objects with metadata preserved
- Use case: Processing loaded documents

### Expected Output

For the sample text, with `chunk_size=100`:
```
[
  'Document loaders in LangChain are responsible for loading, parsing, and preparing data from a wide r',
  'ange of sources such as files, websites, APIs, databases, and cloud storage. These loaders convert r',
  'aw data into a format (typically LangChain `Document` objects) that can be processed for tasks like',
  'embedding, chunking, retrieval, and question-answering.'
]
```

### When to Use
- **Simple text**: When content structure doesn't matter
- **Fixed requirements**: Need exact character counts
- **Testing**: Quick splitting for experimentation
- **Basic RAG**: Simple retrieval systems

### Limitations
- Can break words mid-character
- Ignores sentence boundaries
- No semantic understanding
- May create incomplete thoughts

---

## Tutorial 2: Structure-Based Text Splitting

**File**: `2. text_structure_based.py`

### Concept

`RecursiveCharacterTextSplitter` is smarter than basic character splitting. It tries to split text at natural boundaries (paragraphs, sentences, words) while respecting the chunk size limit.

### How It Works

The splitter uses a hierarchy of separators:
1. **First**: Try to split on double newlines (`\n\n`) - paragraphs
2. **Then**: Try single newlines (`\n`) - lines
3. **Then**: Try spaces (` `) - words
4. **Finally**: Split on characters if necessary

### Code Explanation

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,        # Larger chunks for better context
    chunk_overlap=0,       # No overlap for clear separation
)
```

**Improved Parameters**:
- `chunk_size=200`: More reasonable chunk size
- Uses default separators: `["\n\n", "\n", " ", ""]`

### Sample Text Structure

The example uses text with clear structure:
```
Document loaders in LangChain are responsible for...

The movieId, title, and genres columns will be useful...
```

Notice the double newline (`\n\n`) between paragraphs - the splitter will try to split here first.

### Expected Output

```python
[
  'Document loaders in LangChain are responsible for loading, parsing, and preparing data from a wide range of sources such as files, websites, APIs, databases, and cloud storage. These loaders convert',
  'raw data into a format (typically LangChain `Document` objects) that can be processed for tasks like embedding, chunking, retrieval, and question-answering.',
  'The movieId, title, and genres columns will be useful. You can treat the genres column as the items features (as a list of genres). By transforming this text data into numerical form (e.g., using',
  'one-hot encoding or TF-IDF), you can calculate cosine similarity between items based on their genres. The movieId will connect it with the ratings data frame.'
]
```

### Splitting Logic

1. **First attempt**: Split at `\n\n` (paragraph breaks)
2. **If chunks too big**: Split at `\n` (line breaks)
3. **If still too big**: Split at spaces (word boundaries)
4. **Last resort**: Split at character level

### When to Use
- **Natural text**: Articles, documents, books
- **Preserving structure**: When paragraph/sentence boundaries matter
- **General purpose**: Most common text splitting needs
- **RAG systems**: Better retrieval with meaningful chunks

### Advantages
- Respects natural text structure
- Maintains readability
- Better context preservation
- More intelligent than basic splitting

---

## Tutorial 3: Code-Specific Text Splitting

**File**: `3. code_split.py`

### Concept

`RecursiveCharacterTextSplitter.from_language()` is designed specifically for source code. It understands programming language syntax and splits code at logical boundaries like function definitions, class declarations, and code blocks.

### Language-Aware Splitting

Different programming languages have different structures:
- **Python**: Uses indentation, function/class definitions
- **JavaScript**: Uses braces, function declarations
- **Java**: Uses braces, method/class boundaries
- **And many more**: Each with custom separators

### Code Explanation

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter, Language

splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,    # Optimize for Python syntax
    chunk_size=300,              # Reasonable size for code blocks
    chunk_overlap=0,             # Clean separation
)
```

**Language-Specific Setup**:
- `Language.PYTHON`: Uses Python-aware separators
- Understands function definitions, class declarations
- Respects indentation levels
- Preserves code structure

### Sample Code Structure

The example uses Python code with typical patterns:
- Variable assignments
- Function calls with parameters
- Dictionary structures
- Method chaining

### Python-Specific Separators

For Python, the splitter uses separators like:
```python
[
    "\nclass ",      # Class definitions
    "\ndef ",        # Function definitions
    "\n\tdef ",      # Indented methods
    "\n\n",          # Double newlines
    "\n",            # Single newlines
    " ",             # Spaces
    ""               # Characters
]
```

### Expected Behavior

The splitter will try to:
1. **Keep functions intact**: Avoid splitting mid-function
2. **Preserve indentation**: Maintain Python structure
3. **Split at logical points**: Between functions, classes, or major blocks
4. **Maintain syntax**: Ensure chunks are valid Python when possible

### Supported Languages

Common languages include:
- `Language.PYTHON`
- `Language.JAVASCRIPT`
- `Language.JAVA`
- `Language.CPP`
- `Language.GO`
- `Language.RUST`
- And many more...

### When to Use
- **Code documentation**: Building code search systems
- **Code analysis**: Processing large codebases
- **AI code assistants**: Training or retrieval for coding help
- **Repository indexing**: Making code searchable

### Benefits
- **Syntax awareness**: Understands code structure
- **Logical chunks**: Splits at meaningful boundaries
- **Language optimization**: Tailored for specific languages
- **Maintained functionality**: Chunks more likely to be valid code

---

## Tutorial 4: Semantic-Based Text Splitting

**File**: `4. semantic_bases.py`

### Concept

`SemanticChunker` is the most advanced text splitter. Instead of relying on text structure or character counts, it uses AI embeddings to understand the **meaning** of text and splits when the semantic content changes significantly.

### How Semantic Splitting Works

1. **Generate embeddings**: Each sentence gets converted to a vector
2. **Calculate differences**: Compare semantic similarity between adjacent sentences
3. **Find breakpoints**: When similarity drops below a threshold, create a split
4. **Create chunks**: Group sentences with similar meaning together

### Code Explanation

```python
from langchain_experimental.text_splitter import SemanticChunker
from langchain_ollama import OllamaEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

# Option 1: Using Ollama embeddings
embedding = OllamaEmbeddings(model="mxbai-embed-large")

# Option 2: Using local HuggingFace model
hf_embedding = HuggingFaceEmbeddings(
    model_name=r"E:\LangChain\models\bge-large-en"
)

splitter = SemanticChunker(
    embeddings=embedding,
    breakpoint_threshold_type="standard_deviation",
    breakpoint_threshold_amount=1,
)
```

### Threshold Types

**1. Standard Deviation (Recommended)**:
```python
breakpoint_threshold_type="standard_deviation"
breakpoint_threshold_amount=1  # 1 standard deviation
```
- Splits when difference is 1+ standard deviations from mean
- Good balance of sensitivity and stability
- Works well for most content

**2. Percentile**:
```python
breakpoint_threshold_type="percentile"
breakpoint_threshold_amount=95.0  # 95th percentile
```
- Splits at the top 5% of differences
- More conservative splitting
- Good for very coherent content

**3. Interquartile**:
```python
breakpoint_threshold_type="interquartile"
breakpoint_threshold_amount=1.5  # 1.5 * IQR
```
- Uses interquartile range for outlier detection
- Robust to extreme values
- Good for noisy content

**4. Gradient**:
```python
breakpoint_threshold_type="gradient"
breakpoint_threshold_amount=95.0  # Gradient percentile
```
- Uses gradient + percentile method
- Best for highly semantic content
- Good for legal, medical, technical documents

### Sample Text Analysis

The example text contains three distinct topics:
1. **Farming**: "Farmers were working hard in the fields..."
2. **Cricket**: "The Indian Premier League (IPL) is the biggest..."
3. **Terrorism**: "Terrorism is a big danger to peace..."

### Expected Semantic Chunks

The semantic splitter should identify these topic changes:

```
--- Semantic Chunk 1 ---
Farmers were working hard in the fields, preparing the soil and planting seeds for
the next season. The sun was bright, and the air smelled of earth and fresh grass.

--- Semantic Chunk 2 ---
The Indian Premier League (IPL) is the biggest cricket league in the world. People
all over the world watch the matches and cheer for their favourite teams.

--- Semantic Chunk 3 ---
Terrorism is a big danger to peace and safety. It causes harm to people and creates
fear in cities and villages. When such attacks happen, they leave behind pain and
sadness. To fight terrorism, we need strong laws, alert security forces, and support
from people who care about peace and safety.
```

### Embedding Models

**Ollama Embeddings**:
- Requires Ollama server running
- Model: `mxbai-embed-large`
- Good general-purpose embeddings

**HuggingFace Embeddings**:
- Local model: `bge-large-en`
- No external dependencies
- Consistent performance

### When to Use Semantic Splitting

**Ideal for**:
- **Mixed content**: Documents with multiple topics
- **High-quality retrieval**: When chunk relevance is crucial
- **Complex documents**: Academic papers, reports, articles
- **Domain-specific content**: Legal, medical, technical documents

**Not ideal for**:
- **Simple content**: Uniform topic throughout
- **Performance-critical**: Slower than rule-based splitters
- **Small documents**: Overhead not worth it
- **Structured data**: Code, tables, lists

### Performance Considerations

- **Slower**: Requires embedding calculation for each sentence
- **More accurate**: Better semantic boundaries
- **Resource intensive**: Uses embedding models
- **Quality dependent**: Results depend on embedding model quality

---

## Comparison and Use Cases

### Quick Reference Table

| Splitter Type | Speed | Accuracy | Use Case | Complexity |
|---------------|-------|----------|----------|------------|
| **CharacterTextSplitter** | ⚡⚡⚡ Fast | 🎯 Basic | Simple text, testing | 🔧 Simple |
| **RecursiveCharacterTextSplitter** | ⚡⚡ Fast | 🎯🎯 Good | General documents | 🔧🔧 Medium |
| **Language-Specific** | ⚡⚡ Fast | 🎯🎯🎯 Very Good | Source code | 🔧🔧 Medium |
| **SemanticChunker** | ⚡ Slow | 🎯🎯🎯🎯 Excellent | Complex content | 🔧🔧🔧 Complex |

### Choosing the Right Splitter

**For Simple Text Processing**:
```python
# Quick and dirty splitting
splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
```

**For General Documents**:
```python
# Most common choice - good balance
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
```

**For Source Code**:
```python
# Language-aware splitting
splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON, 
    chunk_size=1000
)
```

**For High-Quality RAG**:
```python
# Semantic understanding
splitter = SemanticChunker(
    embeddings=embeddings,
    breakpoint_threshold_type="standard_deviation"
)
```

## Best Practices

### 1. Chunk Size Guidelines

**For Embeddings**:
- **Small models** (384 dims): 200-500 tokens
- **Large models** (1024+ dims): 500-1000 tokens
- **Very large models**: 1000-2000 tokens

**For RAG Systems**:
- **Question answering**: 200-400 tokens
- **Summarization**: 500-1000 tokens
- **Code retrieval**: 100-300 tokens

### 2. Overlap Considerations

**No Overlap (0)**:
- Clean separation
- No duplicate information
- Good for distinct topics

**Small Overlap (10-20%)**:
- Maintains context
- Good for most use cases
- Balances duplication vs. continuity

**Large Overlap (50%+)**:
- Maximum context preservation
- Higher storage cost
- Good for critical applications

### 3. Parameter Tuning

**Start Conservative**:
```python
# Begin with proven settings
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
```

**Measure Performance**:
- Test retrieval accuracy
- Monitor chunk quality
- Adjust based on results

**Optimize Iteratively**:
- Try different chunk sizes
- Experiment with overlap amounts
- Test with real queries

### 4. Content-Specific Tips

**For Academic Papers**:
- Use semantic splitting
- Larger chunks (800-1200 tokens)
- Preserve paragraph structure

**For Code Documentation**:
- Language-specific splitters
- Smaller chunks (200-400 tokens)
- Preserve function boundaries

**For Mixed Content**:
- Semantic chunking
- Medium chunks (400-800 tokens)
- Higher overlap for context

## Running the Examples

### Prerequisites Setup

1. **Install dependencies**:
   ```bash
   pip install langchain langchain-community langchain-ollama langchain-huggingface langchain-experimental sentence-transformers
   ```

2. **For Ollama embeddings**:
   ```bash
   ollama serve
   ollama pull mxbai-embed-large
   ```

3. **For HuggingFace embeddings**:
   - Download model to `E:\LangChain\models\bge-large-en`
   - Or modify path in code to your model location

### Running Each Tutorial

```bash
# Navigate to text splitters directory
cd "e:\LangChain\8. text_splitters"

# Run tutorials in order
python "1. length_based_split.py"
python "2. text_structure_based.py"
python "3. code_split.py"
python "4. semantic_bases.py"
```

### Troubleshooting

**Import Errors**:
```bash
# Install missing packages
pip install langchain-experimental
```

**Embedding Model Issues**:
```bash
# For Ollama
ollama pull mxbai-embed-large

# For HuggingFace, download model locally or use:
pip install sentence-transformers
```

**Path Issues**:
- Update file paths in code to match your system
- Ensure PDF files exist in referenced locations

### Expected Runtime

- **Length-based**: Instant
- **Structure-based**: Instant  
- **Code splitting**: Instant
- **Semantic**: 5-30 seconds (depending on text length and model)

## Summary

Text splitters are fundamental tools for working with large documents in LangChain:

### Key Takeaways

1. **Start Simple**: Use `RecursiveCharacterTextSplitter` for most cases
2. **Consider Content**: Use appropriate splitter for your content type
3. **Optimize Iteratively**: Tune parameters based on performance
4. **Balance Speed vs. Quality**: Choose complexity level based on needs

### Progression Path

1. **Learn**: Start with basic character splitting
2. **Improve**: Move to structure-aware splitting
3. **Specialize**: Use language-specific for code
4. **Optimize**: Apply semantic splitting for best results

### Production Recommendations

- **Development**: Use recursive character splitter
- **Testing**: Measure chunk quality and retrieval performance
- **Production**: Choose based on speed/quality requirements
- **Optimization**: Consider semantic splitting for critical applications

The tutorials progress from simple to sophisticated, giving you the tools to handle any text splitting challenge in your LangChain applications.
