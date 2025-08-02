from langchain.prompts import PromptTemplate

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
input_variables = ["paper_name", "explanation_type", "length"]
validate_template = True

template_instance = PromptTemplate(
    template=template,
    input_variables=input_variables,
    validate_template=validate_template
)

template_instance.save("dynamic_prompt_template.json")
