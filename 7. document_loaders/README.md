# Document Loaders Tutorial

This tutorial shows you how to load different types of documents in LangChain and process them with AI models.

## Files Overview
- `1. text_loader.py` - Load text files and generate summaries
- `2. pypdf_loader.py` - Extract text from PDF files
- `3. directory_loader.py` - Load multiple PDF files at once
- `4. lazy_loader_Vs_loader.py` - Memory-efficient loading strategies
- `5. csv_loader.py` - Process CSV data with AI
- `5. web_based_loader.py` - Load content from web pages

---

## Tutorial 1: Text File Loading (`1. text_loader.py`)

### What it does:
Loads a text file and creates an AI summary using ChatOllama.

### Step-by-step breakdown:

**Step 1: Import libraries**
```python
from langchain_community.document_loaders import TextLoader
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import time
```

**Step 2: Setup AI model**
```python
model = ChatOllama(
    model="gemma2:2b",
    temperature=0.5,
)
```
- Uses gemma2:2b model (lightweight and fast)
- temperature=0.5 balances creativity and consistency

**Step 3: Create prompt template**
```python
prompt = PromptTemplate(
    template="Write the summary of the following document: {text}",
    input_variables=["text"],
)
```
- Template asks AI to summarize the document
- {text} will be replaced with actual document content

**Step 4: Setup output parser**
```python
parser = StrOutputParser()
```
- Converts AI response to clean string

**Step 5: Load the document**
```python
loader = TextLoader(r"E:\LangChain\7. document_loaders\Documents\Langchain DocumentLoader Types.txt", encoding="utf-8")
document = loader.load()
```
- Loads text file with UTF-8 encoding
- Returns list with one Document object

**Step 6: Create processing chain**
```python
chain = prompt | model | parser
```
- Combines prompt → model → parser in sequence

**Step 7: Process and measure time**
```python
start = time.time()
result = chain.invoke({"text": document[0].page_content})
print(result)
end = time.time()
print(f"Processing time: {end - start} seconds")
```
- Processes document content through the chain
- Measures and displays processing time

**Step 8: Show document info**
```python
print(len(document))      # Number of documents (1)
print(type(document))     # List
print(type(document[0]))  # Document object
print(document[0])        # Full document with metadata
```

---

## Tutorial 2: PDF Loading (`2. pypdf_loader.py`)

### What it does:
Extracts text from PDF files, with each page as a separate document.

### Step-by-step breakdown:

**Step 1: Import libraries**
```python
from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import time
```

**Step 2: Set document path**
```python
document_path = r"E:\LangChain\7. document_loaders\Documents\Movie Recommend Approach.pdf"
```

**Step 3: Load PDF**
```python
loader = PyPDFLoader(document_path)
document = loader.load()
```
- Each PDF page becomes a separate Document object

**Step 4: Analyze results**
```python
print(len(document))                # Number of pages
print(document[0].page_content)     # First page content
```

### Key Points:
- PDF with 10 pages = 10 Document objects
- Each document has page content and metadata (source file, page number)
- Good for multi-page documents like research papers

---

## Tutorial 3: Directory Loading (`3. directory_loader.py`)

### What it does:
Loads all PDF files from a directory at once using pattern matching.

### Step-by-step breakdown:

**Step 1: Import libraries**
```python
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
```

**Step 2: Setup directory loader**
```python
docs = DirectoryLoader(
    path=r"E:\LangChain\7. document_loaders\Documents",
    glob="**/*.pdf",         # Find all PDF files, including subdirectories
    loader_cls=PyPDFLoader,  # Use PyPDFLoader for each PDF
)
```

**Step 3: Load all documents**
```python
lists = docs.load()
```
- Scans directory for PDF files
- Loads each PDF using PyPDFLoader
- Combines all pages into one list

**Step 4: Check results**
```python
print(len(lists))                   # Total pages from all PDFs
print(lists[20].page_content)       # Content from page 21
```

### Glob Patterns:
- `*.pdf` = PDFs in current directory only
- `**/*.pdf` = PDFs in all subdirectories too
- `**/*.txt` = All text files everywhere

---

## Tutorial 4: Memory Management (`4. lazy_loader_Vs_loader.py`)

### What it does:
Compares two loading strategies for memory efficiency.

### Step-by-step breakdown:

**Step 1: Setup (same as directory loader)**
```python
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader

docs = DirectoryLoader(
    path=r"E:\LangChain\7. document_loaders\Documents",
    glob="**/*.pdf",
    loader_cls=PyPDFLoader,
)
```

**Step 2: Lazy Loading (Memory Efficient)**
```python
lists = docs.lazy_load()
for i in lists:
    print(i.metadata)  # Load one document at a time
```
- Documents loaded one by one
- Previous document freed from memory
- Good for large collections

**Step 3: Regular Loading (Memory Intensive)**
```python
lists = docs.load()
for i in lists:
    print(i.metadata)  # All documents already in memory
```
- All documents loaded at once
- All stay in memory until finished
- Fast access but uses more RAM

### When to use:
- **Lazy loading**: Large directories, limited RAM
- **Regular loading**: Small collections, need speed

---

## Tutorial 5: CSV Processing (`5. csv_loader.py`)

### What it does:
Loads CSV data and analyzes it with AI.

### Step-by-step breakdown:

**Step 1: Import libraries**
```python
from langchain_community.document_loaders import CSVLoader
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
```

**Step 2: Setup AI model**
```python
chat = ChatOllama(
    model="gemma3:1b",
    temperature=0.5
)
parser = StrOutputParser()
```

**Step 3: Create analysis prompt**
```python
prompt = PromptTemplate(
    template="Extract the first 5 rows from the CSV file: {file} \n\nand explain what the data represents.",
    input_variables=["file"],
)
```

**Step 4: Load CSV file**
```python
document_loader = CSVLoader(
    file_path=r"E:\LangChain\7. document_loaders\Documents\Student Mental health.csv",
)
documents = document_loader.load()
```
- Each CSV row becomes a Document object

**Step 5: Process first 5 rows**
```python
chain = prompt | chat | parser

first_5 = "\n\n".join([doc.page_content for doc in documents[:5]])
print(first_5)
print(chain.invoke({"file": first_5}))
```
- Combines first 5 rows
- Asks AI to explain the data

---

## Tutorial 6: Web Loading (`5. web_based_loader.py`)

### What it does:
Loads content from web pages using two different methods.

### Step-by-step breakdown:

**Step 1: Import libraries**
```python
from langchain_community.document_loaders import PlaywrightURLLoader, WebBaseLoader
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
import time
```

**Step 2: Setup AI model and prompt**
```python
chat = ChatOllama(
    model="gemma3:1b",
    temperature=0.5
)

prompt = PromptTemplate(
    template="Summarize the product details from the following content: {content}",
    input_variables=["content"]
)

parser = StrOutputParser()
```

**Step 3: Define URLs**
```python
urls = ["https://www.applegadgetsbd.com/product/apple-mac-mini-m4-10-core-cpu-10-core-gpu-16-256gb"]  # For Playwright
url = "https://en.wikipedia.org/wiki/Pauline_Ferrand-Pr%C3%A9vot"  # For WebBase
```

**Step 4: Load with WebBaseLoader (Static pages)**
```python
loader_web = WebBaseLoader(url)
documents_web = loader_web.load()
```
- Fast for simple websites
- Good for static content

**Step 5: Load with PlaywrightURLLoader (Dynamic pages)**
```python
loader = PlaywrightURLLoader(urls=urls, headless=True)
documents = loader.load()
```
- Handles JavaScript
- Good for complex websites
- Requires: `playwright install`

**Step 6: Process with AI**
```python
chain = prompt | chat | parser

begin = time.time()
result = chain.invoke({"content": documents[0].page_content})
end = time.time()

print(result)
print(f"Time taken: {end - begin} seconds")
```

---

## Installation Guide

**Install required packages:**
```bash
pip install langchain-community langchain-ollama langchain-core
```

**For web loading:**
```bash
pip install playwright
playwright install
```

**Setup Ollama models:**
```bash
ollama pull gemma2:2b
ollama pull gemma3:1b
```

## Running Examples

```bash
cd "E:\LangChain\7. document_loaders"
python "1. text_loader.py"
python "2. pypdf_loader.py"
python "3. directory_loader.py"
python "4. lazy_loader_Vs_loader.py"
python "5. csv_loader.py"
python "5. web_based_loader.py"
```

## Quick Reference

| File Type | Loader | Best For |
|-----------|---------|----------|
| .txt | TextLoader | Simple text files |
| .pdf | PyPDFLoader | PDF documents |
| Multiple files | DirectoryLoader | Batch processing |
| .csv | CSVLoader | Spreadsheet data |
| Web pages | WebBaseLoader/PlaywrightURLLoader | Online content |

## Tips

1. **Memory**: Use lazy loading for large collections
2. **Encoding**: Add `encoding="utf-8"` for text files
3. **Paths**: Use raw strings `r"path"` for Windows paths
4. **Testing**: Start with small files first
5. **Error handling**: Check if documents loaded before processing