text='''
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
'''

from langchain.text_splitter import RecursiveCharacterTextSplitter,Language

splitter= RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=300,
    chunk_overlap=0,
)

result=splitter.split_text(text=text)

print(result[0])