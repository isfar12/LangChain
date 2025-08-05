from langchain_core.runnables import RunnableParallel, RunnableSequence
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
    template="Write me a Linkedin medium length post about {topic}, Just give me the post, no explanation.",
    input_variables=["topic"]
)


prompt2 = PromptTemplate(
    template="Write me a tweet about {topic}, just give me the tweet, no explanation.",
    input_variables=["topic"]
)

parser = StrOutputParser()

# Create parallel runnables by chaining them together sequentially one after another
parallel_chain = RunnableParallel(
   { 
    "linkedin_post": RunnableSequence(prompt1, model1, parser),
    "tweet": RunnableSequence(prompt2, model2, parser)
    }
)

result= parallel_chain.invoke({"topic": "AI and its impact on society"})

print(result["linkedin_post"])
print(result["tweet"])