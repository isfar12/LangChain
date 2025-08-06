# Document Loaders in LangChain

## Overview

Document loaders are essential components in LangChain that help you import and process various types of documents into a format that can be used by language models. They convert different file formats (text, PDF, CSV, web pages) into LangChain Document objects that contain both content and metadata.

## What is a Document Loader?

A document loader is a tool that:
- Reads files from different sources (local files, URLs, databases)
- Converts content into LangChain Document objects
- Preserves metadata (file path, page numbers, etc.)
- Handles different file formats automatically

## Document Structure

When a document loader processes a file, it creates Document objects with:
- **page_content**: The actual text content
- **metadata**: Information about the source (file path, page number, etc.)

## Available Document Loaders

### 1. Text Loader (`1. text_loader.py`)

**Purpose**: Loads plain text files (.txt) and processes them with AI models.

**Key Features**:
- Handles text encoding (UTF-8, etc.)
- Loads entire text file as a single document
- Simple and fast for basic text processing

**Code Breakdown**:
```python
from langchain_community.document_loaders import TextLoader

# Create loader with encoding specification
loader = TextLoader(
    r"E:\LangChain\7. document_loaders\Documents\Langchain DocumentLoader Types.txt", 
    encoding="utf-8"
)

# Load document - returns list of Document objects
document = loader.load()
```

**What happens**:
1. TextLoader reads the entire file
2. Creates one Document object with all content
3. Stores file path in metadata
4. Content is accessible via `document[0].page_content`

**When to use**: 
- Plain text files
- Configuration files
- Simple documents without complex formatting

---

### 2. PDF Loader (`2. pypdf_loader.py`)

**Purpose**: Extracts text from PDF files, handling multiple pages automatically.

**Key Features**:
- Splits PDF into individual pages
- Each page becomes a separate Document object
- Preserves page numbers in metadata
- Works with most standard PDFs

**Code Breakdown**:
```python
from langchain_community.document_loaders import PyPDFLoader

# Create PDF loader
loader = PyPDFLoader(document_path)

# Load document - returns list with one Document per page
document = loader.load()
```

**What happens**:
1. PyPDFLoader opens the PDF file
2. Extracts text from each page separately
3. Creates Document object for each page
4. Metadata includes page number and source file

**Document structure**:
- `document[0]` = Page 1 content
- `document[1]` = Page 2 content
- Each has metadata: `{'source': 'file_path', 'page': page_number}`

**When to use**:
- Academic papers
- Reports
- Books
- Any multi-page PDF documents

---

### 3. Directory Loader (`3. directory_loader.py`)

**Purpose**: Loads multiple files from a directory using pattern matching.

**Key Features**:
- Processes multiple files at once
- Uses glob patterns to filter files
- Applies specified loader to each matching file
- Combines all documents into one list

**Code Breakdown**:
```python
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

docs = DirectoryLoader(
    path=r"E:\LangChain\7. document_loaders\Documents",
    glob="**/*.pdf",  # Find all PDF files, including subdirectories
    loader_cls=PyPDFLoader  # Use PyPDFLoader for each PDF
)

documents = docs.load()
```

**What happens**:
1. Scans directory for files matching pattern
2. Applies PyPDFLoader to each PDF found
3. Combines all resulting documents into one list
4. Each document retains its source file information

**Glob patterns**:
- `"*.pdf"` - All PDFs in current directory
- `"**/*.pdf"` - All PDFs in current and subdirectories
- `"**/*.txt"` - All text files everywhere
- `"data/*.csv"` - All CSVs in data folder

**When to use**:
- Processing multiple files of same type
- Batch document processing
- Building document databases

---

### 4. Lazy Loading vs Regular Loading (`4. lazy_loader_Vs_loader.py`)

**Purpose**: Demonstrates two different loading strategies for memory management.

**Regular Loading (`.load()`)**:
- Loads ALL documents into memory immediately
- Fast access to all documents
- High memory usage
- Good for small datasets

**Lazy Loading (`.lazy_load()`)**:
- Loads documents one at a time as needed
- Lower memory usage
- Slower access
- Good for large datasets

**Code Breakdown**:
```python
# Lazy loading - memory efficient
lazy_docs = docs.lazy_load()
for doc in lazy_docs:
    print(doc.metadata)  # Loads one document at a time

# Regular loading - loads everything
all_docs = docs.load()
for doc in all_docs:
    print(doc.metadata)  # All documents already in memory
```

**When to use lazy loading**:
- Large directories with many files
- Limited memory environments
- Processing documents one at a time

**When to use regular loading**:
- Small datasets
- Need random access to documents
- Performing multiple operations on same data

---

### 5. CSV Loader (`5. csv_loader.py`)

**Purpose**: Loads CSV files and converts each row into a separate document.

**Key Features**:
- Each CSV row becomes a Document object
- Column headers are preserved
- Handles different CSV formats
- Good for structured data processing

**Code Breakdown**:
```python
from langchain_community.document_loaders import CSVLoader

document_loader = CSVLoader(
    file_path=r"E:\LangChain\7. document_loaders\Documents\Student Mental health.csv"
)

documents = document_loader.load()
```

**What happens**:
1. Reads CSV file with headers
2. Converts each row to a Document
3. Row content becomes page_content
4. File information stored in metadata

**Document format**:
Each row becomes:
```
page_content: "column1: value1\ncolumn2: value2\ncolumn3: value3"
metadata: {'source': 'file_path', 'row': row_number}
```

**When to use**:
- Customer data
- Survey responses
- Tabular data analysis
- Structured datasets

---

### 6. Web-Based Loader (`5. web_based_loader.py`)

**Purpose**: Loads content from web pages using two different approaches.

**Two Types**:

**WebBaseLoader**:
- For static web pages
- Fast and lightweight
- Good for simple HTML content
- Limited JavaScript support

**PlaywrightURLLoader**:
- For dynamic web pages
- Full browser simulation
- Handles JavaScript
- Slower but more capable

**Code Breakdown**:
```python
from langchain_community.document_loaders import PlaywrightURLLoader, WebBaseLoader

# For static pages
loader_web = WebBaseLoader(url)
documents_web = loader_web.load()

# For dynamic pages
loader = PlaywrightURLLoader(urls=urls, headless=True)
documents = loader.load()
```

**WebBaseLoader use cases**:
- News articles
- Blog posts
- Static product pages
- Simple websites

**PlaywrightURLLoader use cases**:
- E-commerce sites with dynamic content
- Social media pages
- JavaScript-heavy applications
- Sites requiring interaction

---

## Sample Documents

The `Documents/` folder contains example files for testing:

- **Langchain DocumentLoader Types.txt**: Text file with documentation
- **Movie Recommend Approach.pdf**: PDF document for testing
- **Student Mental health.csv**: CSV data for analysis
- **Software Engineering Overview.pdf**: Multi-page PDF
- **Scanned Optics PDF.pdf**: Another PDF sample

## Common Patterns

### Basic Loading Pattern
```python
# 1. Import loader
from langchain_community.document_loaders import LoaderClass

# 2. Create loader instance
loader = LoaderClass(file_path_or_parameters)

# 3. Load documents
documents = loader.load()

# 4. Access content
content = documents[0].page_content
metadata = documents[0].metadata
```

### Processing with AI Models
```python
# After loading documents
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate

model = ChatOllama(model="gemma2:2b")
prompt = PromptTemplate(
    template="Summarize: {text}",
    input_variables=["text"]
)

# Process document content
result = (prompt | model).invoke({"text": documents[0].page_content})
```

## Error Handling

Common issues and solutions:

**File not found**:
```python
try:
    documents = loader.load()
except FileNotFoundError:
    print("File not found - check file path")
```

**Encoding issues**:
```python
# Specify encoding for text files
loader = TextLoader(file_path, encoding="utf-8")
```

**Memory issues with large files**:
```python
# Use lazy loading
documents = loader.lazy_load()
for doc in documents:
    process_document(doc)
```

## Performance Tips

1. **Use appropriate loader**: Don't use PDF loader for text files
2. **Lazy loading**: For large datasets to save memory
3. **Batch processing**: Process multiple similar documents together
4. **Caching**: Store processed results to avoid reloading
5. **File size**: Break very large files into smaller chunks

## Best Practices

1. **Always check document structure**: Print first document to understand format
2. **Handle metadata**: Use metadata for document tracking and filtering
3. **Validate content**: Check if content loaded correctly before processing
4. **Error handling**: Implement proper error handling for file operations
5. **Resource management**: Clean up resources, especially with web loaders

## Running the Examples

To run any example:

1. **Install requirements**:
   ```bash
   pip install langchain-community langchain-ollama langchain-core
   ```

2. **For web loader (Playwright)**:
   ```bash
   pip install playwright
   playwright install
   ```

3. **Start Ollama**:
   ```bash
   ollama serve
   ollama pull gemma2:2b
   ollama pull gemma3:1b
   ```

4. **Run examples**:
   ```bash
   cd "e:\LangChain\7. document_loaders"
   python "1. text_loader.py"
   python "2. pypdf_loader.py"
   # ... and so on
   ```

## Summary

Document loaders are the foundation of document processing in LangChain. They provide:

- **Unified interface** for different file types
- **Automatic content extraction** from various formats
- **Metadata preservation** for document tracking
- **Memory management** options for different use cases
- **Web content access** for dynamic data sources

Choose the right loader based on your file type and processing needs. Start with simple loaders for basic use cases and move to more specialized ones as requirements grow.
