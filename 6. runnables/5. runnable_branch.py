from langchain_core.runnables import RunnableSequence, RunnableBranch, RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama

# Define the prompt template
model= ChatOllama(model="llama3.1", temperature=0.5)

prompt = PromptTemplate(
    template="Write a report on the following topic: {topic}",
    input_variables=["topic"],
)

summarize_prompt = PromptTemplate(
    template="Summarize the following report within 150 words: {text}",
    input_variables=["text"],
)

parser= StrOutputParser()

report_generator = RunnableSequence(prompt, model, parser)

branch_chain = RunnableBranch(
    (lambda x:len(x.split())>250, RunnableSequence(summarize_prompt, model, parser)), # if the report is long, summarize it
    RunnablePassthrough() # if the report is short enough, just pass it through and dont summarize
)

final_chain = RunnableSequence(report_generator, branch_chain)

print(final_chain.invoke({"topic": "The impact of AI on modern society"}))
