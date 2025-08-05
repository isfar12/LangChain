from langchain_core.runnables import RunnablePassthrough, RunnableSequence, RunnableParallel
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


model1 = ChatOllama(
    model="gemma3:1b",
    temperature=0.7
)


model2 = ChatOllama(
    model="gemma2:2b",
    temperature=0.7
)

prompt1 = PromptTemplate(
    template="Write me a Linkedin post about {topic}, Just give me the post, no explanation.",
    input_variables=["topic"]
)


prompt2 = PromptTemplate(
    template="Write summary about the post using points. Here is the post provided: {topic}.",
    input_variables=["topic"]
)

parser = StrOutputParser()

linkedin_generator = RunnableSequence(prompt1, model1, parser)

# Create parallel runnables 
parallel_chain = RunnableParallel(
   { 
    "linkedin_post": RunnablePassthrough(), # This will pass the input through without any modification
    "summary": RunnableSequence(prompt2, model2, parser) # This will generate a summary of the post
    }
)

final_chain = RunnableSequence(linkedin_generator, parallel_chain) # we will have final output as a dictionary with keys linkedin_post and summary

print(final_chain.invoke({"topic": "AI and its impact on society"}))
