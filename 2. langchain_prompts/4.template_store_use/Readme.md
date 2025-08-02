# LangChain Prompt Template Storage and Usage

This folder demonstrates how to create, store, and reuse prompt templates in LangChain. It showcases dynamic prompt customization with persistent storage and interactive user interfaces.

## 📁 Folder Contents

```
template_store_use/
├── template_gen.py              # Template creation and storage
├── dynamic_prompt_template.json # Saved template file
├── new_prompt_ui.py            # Streamlit UI for template usage
└── Readme.md                   # This documentation
```

## 🎯 Purpose

This project demonstrates:
1. **Template Creation**: How to create structured prompt templates
2. **Template Storage**: Saving templates to JSON files for reuse
3. **Template Loading**: Loading saved templates in different applications
4. **Dynamic Customization**: Creating interactive UIs for prompt customization
5. **Best Practices**: Professional prompt engineering techniques

## 📝 File Explanations

### 1. `template_gen.py` - Template Generator

**Purpose**: Creates and saves a reusable prompt template for paper explanation tasks.

#### Code Breakdown:

```python
from langchain.prompts import PromptTemplate

# Define the template structure
template = """
Please provide a detailed explanation of the following paper:

Paper Name: {paper_name}
Explanation Type: {explanation_type}
Output Length: {length}
1. If mathematical equations are present, explain them in simple terms.
2. Provide a beginner-friendly overview of the paper.
3. Include a technical summary for advanced readers.
4. Use related examples to illustrate key concepts.

Ensure the explanation is clear and concise, tailored to the selected output length.
"""
```

#### Key Components:

**Template Variables:**
- `{paper_name}`: Dynamic paper title input
- `{explanation_type}`: Type of explanation (summary, detailed, etc.)
- `{length}`: Desired output length (short, medium, long)

**Template Features:**
- **Structured Instructions**: Clear, numbered guidelines
- **Flexibility**: Adaptable to different papers and requirements
- **User-Friendly**: Beginner-friendly explanations
- **Technical Depth**: Advanced summaries for experts

**Template Creation:**
```python
template_instance = PromptTemplate(
    template=template,
    input_variables=["paper_name", "explanation_type", "length"],
    validate_template=True  # Ensures template integrity
)

# Save for reuse
template_instance.save("dynamic_prompt_template.json")
```

**Benefits:**
- ✅ **Reusability**: Save once, use everywhere
- ✅ **Consistency**: Standardized format across applications
- ✅ **Validation**: Built-in template validation
- ✅ **Version Control**: JSON format is git-friendly

---

### 2. `dynamic_prompt_template.json` - Stored Template

**Purpose**: Persistent storage of the prompt template in JSON format.

#### Structure:
```json
{
    "name": null,
    "input_variables": [
        "explanation_type",
        "length",
        "paper_name"
    ],
    "template": "Please provide a detailed explanation of the following paper:\n\nPaper Name: {paper_name}\n..."
}
```

**Key Features:**
- **Serialization**: Template stored as structured data
- **Portability**: Can be shared across projects and teams
- **Version Control**: Easy to track changes
- **Language Agnostic**: JSON can be used in any programming language

---

### 3. `new_prompt_ui.py` - Interactive Streamlit Interface

**Purpose**: Provides a user-friendly web interface for using the stored template with dynamic inputs.

#### Key Components:

**Model Setup:**
```python
from langchain_ollama import ChatOllama
model = ChatOllama(model="phi3", temperature=0.5)
```

**Template Loading:**
```python
from langchain_core.prompts import load_prompt
# Loads the saved template
template = load_prompt("dynamic_prompt_template.json")
```

**Interactive UI Elements:**
```python
import streamlit as st

st.header("Dynamic Prompt Customization")

# Dropdown for paper selection
paper_name = st.selectbox("Select a Paper", [...])

# Other dynamic inputs for explanation_type and length
```

**Features:**
- 🎨 **User-Friendly Interface**: Web-based interaction
- 🔄 **Dynamic Inputs**: Real-time prompt customization
- 📊 **Interactive Elements**: Dropdowns, text inputs, sliders
- 🚀 **Live Generation**: Immediate AI responses

---

## 🛠️ How It Works

### Workflow Overview:

```
1. Template Creation (template_gen.py)
   ↓
2. Template Storage (dynamic_prompt_template.json)
   ↓
3. Template Loading & Usage (new_prompt_ui.py)
   ↓
4. Dynamic Prompt Generation
   ↓
5. AI Model Response
```

### Step-by-Step Process:

1. **Create Template** (`template_gen.py`):
   - Define prompt structure with variables
   - Add instructions and formatting
   - Validate template syntax
   - Save to JSON file

2. **Store Template** (`dynamic_prompt_template.json`):
   - JSON serialization of template
   - Persistent storage for reuse
   - Version control friendly

3. **Load & Use** (`new_prompt_ui.py`):
   - Load saved template
   - Create interactive interface
   - Generate dynamic prompts
   - Get AI responses

## 🎯 Use Cases

### 1. **Academic Paper Analysis**
- **Input**: Paper name, explanation type, length
- **Output**: Structured paper explanation
- **Users**: Students, researchers, academics

### 2. **Content Summarization**
- **Adaptable for**: Articles, reports, documents
- **Benefits**: Consistent formatting, customizable depth

### 3. **Educational Tools**
- **Use**: Teaching complex concepts
- **Features**: Beginner-friendly explanations, technical summaries

### 4. **Research Assistance**
- **Purpose**: Quick paper overviews
- **Advantage**: Standardized analysis format

## 🚀 Running the Applications

### Prerequisites:
```bash
pip install langchain langchain-ollama streamlit
```

### 1. Generate Template:
```bash
python template_gen.py
```
*Creates `dynamic_prompt_template.json`*

### 2. Run Interactive UI:
```bash
streamlit run new_prompt_ui.py
```
*Opens web interface at `http://localhost:8501`*

## 🔧 Customization Options

### Template Modifications:

1. **Add New Variables**:
```python
input_variables = ["paper_name", "explanation_type", "length", "audience"]
```

2. **Modify Instructions**:
```python
template = """
Your custom instructions here with {variables}
"""
```

3. **Change Output Format**:
```python
template = """
Format: {format_type}
Style: {writing_style}
Content: {content_focus}
"""
```

### UI Customization:

1. **Add New Input Types**:
```python
# Slider for complexity level
complexity = st.slider("Complexity Level", 1, 10, 5)

# Text area for custom instructions
custom_notes = st.text_area("Additional Notes")

# Multiple select for topics
topics = st.multiselect("Focus Areas", ["Math", "Methodology", "Results"])
```

2. **Styling Options**:
```python
st.markdown("## Custom Styling")
st.info("Information boxes")
st.warning("Warning messages")
st.success("Success indicators")
```

## 💡 Best Practices Demonstrated

### 1. **Template Design**:
- ✅ Clear, structured instructions
- ✅ Flexible variable placement
- ✅ Comprehensive guidance
- ✅ User-centric approach

### 2. **Code Organization**:
- ✅ Separation of concerns
- ✅ Reusable components
- ✅ Clean file structure
- ✅ Proper documentation

### 3. **User Experience**:
- ✅ Intuitive interface design
- ✅ Real-time feedback
- ✅ Error handling
- ✅ Progressive disclosure

### 4. **Maintenance**:
- ✅ Version control friendly
- ✅ Easy to modify
- ✅ Shareable templates
- ✅ Scalable architecture

## 🔍 Advanced Features

### Template Composition:
```python
# Combine multiple templates
base_template = load_prompt("base_template.json")
specific_template = load_prompt("specific_template.json")
```

### Conditional Logic:
```python
# Add conditional instructions based on input
if explanation_type == "technical":
    template += "\nInclude mathematical formulations and proofs."
```

### Template Inheritance:
```python
# Create template hierarchies
class PaperExplanationTemplate(PromptTemplate):
    def __init__(self):
        super().__init__(...)
```

## 🆘 Troubleshooting

### Common Issues:

1. **Template Validation Errors**:
   - Check variable names match exactly
   - Ensure all variables are defined
   - Verify JSON syntax

2. **Streamlit Issues**:
   - Install correct dependencies
   - Check port availability
   - Verify file paths

3. **Model Connection**:
   - Ensure Ollama is running
   - Check model availability
   - Verify network settings

## 📚 Learning Outcomes

After working with this folder, you'll understand:

1. **Prompt Engineering**: How to create effective, reusable prompts
2. **Template Management**: Storing and loading templates efficiently
3. **UI Development**: Building interactive interfaces for AI applications
4. **Best Practices**: Professional approaches to prompt template design
5. **Integration**: Combining different LangChain components effectively

## 🔗 Related Concepts

- **Prompt Engineering**: Crafting effective AI prompts
- **Template Patterns**: Reusable design patterns for prompts
- **UI/UX Design**: Creating user-friendly AI interfaces
- **Configuration Management**: Storing and managing application settings
- **Modular Architecture**: Building maintainable AI applications

This folder serves as a comprehensive example of professional prompt template management in LangChain applications.
