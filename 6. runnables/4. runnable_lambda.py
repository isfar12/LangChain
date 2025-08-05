from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnableLambda, RunnableSequence, RunnableParallel
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

chat = ChatOllama(
    model="gemma3:1b",
    temperature=0.7
)

prompt1 = PromptTemplate(
    template="Write me a Linkedin post about {topic}, Just give me the post, no explanation.",
    input_variables=["topic"]
)

parser = StrOutputParser()

# RunnableLambda is used to create a custom function that can be used in the runnable chain to modify or process the output of other runnables.
def count_words(text):
    return len(text.split())


# Create a runnable that generates a Linkedin post and counts the number of words in the post

linkedin_generator = RunnableSequence(prompt1, chat, parser)

parallel_chain = RunnableParallel(
    { 
        "linkedin_post": linkedin_generator,
        "word_count": RunnableLambda(count_words)  # This will count the words in the post
    }
)

final_chain = RunnableSequence(linkedin_generator,parallel_chain)
result = final_chain.invoke({"topic": "AI and its impact on society"})

print(result)