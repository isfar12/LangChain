from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableBranch,RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser,StrOutputParser
from pydantic import BaseModel, Field
from typing import Literal


# define strucuted output
class Sentiment(BaseModel):
    sentiment: Literal["positive", "negative"] = Field(..., description="The sentiment of the feedback.")

# pydantics output parser ensures to get structured output
parser= PydanticOutputParser(pydantic_object=Sentiment)

# parser output parser ensures to get string output
parser_out = StrOutputParser()

# Define the prompt template with the output parser
prompt = PromptTemplate(
    template="Write only the classification of the sentiment of the following feedback text as either 'positive' or 'negative': \n{feedback} \n{formatted_instructions}",
    input_variables=["feedback"],
    partial_variables={"formatted_instructions": parser.get_format_instructions()},
)

chat=ChatOllama(
    model="llama3.1",
    temperature=0.5,
)
# Create a chain that uses the prompt and the chat model

chain = prompt | chat | parser


prompt1 = PromptTemplate(
    template="Write only one feedback response to the following sentiment: \n{feedback}",
    input_variables=["feedback"],
)
prompt2 = PromptTemplate(
    template="Write only one feedback response to the following sentiment: \n{feedback}",
    input_variables=["feedback"],
)
# our output will look like this:
# sentiment='negative'

runnable_branch = RunnableBranch(
    # format (condition, runnable)
    (lambda x:x.sentiment == "negative", prompt1 | chat | parser),
    (lambda x:x.sentiment == "positive", prompt2 | chat | parser_out),
    RunnableLambda(lambda x: "Could not find sentiment")
)

chain_with_branch = chain | runnable_branch
chain_with_branch.get_graph().print_ascii()
result= chain_with_branch.invoke(
    {
        "feedback": "I love the new features in the latest update, they are fantastic!"
    }
)

print(result)  
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
+----------------------+
| PydanticOutputParser |
+----------------------+
            *
            *
            *
       +--------+
       | Branch |
       +--------+
            *
            *
            *
    +--------------+
    | BranchOutput |
    +--------------+
'''