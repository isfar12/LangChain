from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama

chat=ChatOllama(
    model="gemma3:1b",
    temperature=0.5,
)

prompt=PromptTemplate(
    template="Write a report on {topic}",
    input_variables=["topic"],
)

next_prompt= PromptTemplate(
    template="Write a short summary of the report on {topic}",
    input_variables=["topic"],
)

parser=StrOutputParser()
chain = prompt | chat | parser | next_prompt | chat | parser

chain.get_graph().print_ascii()

print(chain.invoke({"topic": "the impact of AI on society"}))

#  Model graph structure:
'''
     +-------------+       
     | PromptInput |       
     +-------------+       
            *
            *
            *
    +----------------+     
    | PromptTemplate |     
    +----------------+     
            *
            *
            *
      +------------+       
      | ChatOllama |       
      +------------+       
            *
            *
            *
   +-----------------+
   | StrOutputParser |
   +-----------------+
            *
            *
            *
+-----------------------+
| StrOutputParserOutput |
+-----------------------+
            *
            *
            *
    +----------------+
    | PromptTemplate |
    +----------------+
            *
            *
            *
      +------------+
      | ChatOllama |
      +------------+
            *
            *
            *
   +-----------------+
   | StrOutputParser |
   +-----------------+
            *
            *
            *
+-----------------------+
| StrOutputParserOutput |
+-----------------------+
'''