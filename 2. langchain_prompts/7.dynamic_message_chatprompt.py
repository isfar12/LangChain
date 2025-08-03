from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

model = ChatOllama(model="gemma:2b", temperature=0.7)

chat_template = ChatPromptTemplate.from_messages( # we can skip from_messages 
    messages=[ # instead of using the class, we just use the names like this and pass the content
       ("system", "You are a helpful assistant in the field of {domain}."),
         ("human", "Explain the concept of {concept} in simple terms."),
    ]
)

result=chat_template.invoke({
    "domain": "science",
    "concept": "photosynthesis"
})
print(result)
