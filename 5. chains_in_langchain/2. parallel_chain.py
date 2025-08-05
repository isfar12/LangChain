from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

chat1 = ChatOllama(
    model="gemma3:1b",
    temperature=0.5,
)
chat2 = ChatOllama(
    model="llama3.2:latest",
    temperature=0.2,
)

prompt1= PromptTemplate(
    template="Generate short and simple notes from the following text: \n S {topic}",
    input_variables=["topic"],
)

prompt2 = PromptTemplate(
    template="Write 5 questions about the notes from the information on {topic}",
    input_variables=["topic"],
)

prompt3= PromptTemplate(
    template="Merge the provided notes and questions into a single document \n the notes are: {notes} \n the questions are: {questions}",
    input_variables=["notes", "questions"],
)

parser = StrOutputParser( )

parallel_chain = RunnableParallel(
    {
        "notes": prompt1 | chat1 | parser,
        "questions": prompt2 | chat2 | parser,
    }
)


merged_chain = prompt3 | chat1 | parser

final_output =parallel_chain | merged_chain

final_output.get_graph().print_ascii() # print the graph structure

topic ='''
The Kaggle community has a lot of diversity, with members from over 100 countries and skill levels ranging from those learning Python through to the researchers who created deep neural networks. We have had competition winners with backgrounds ranging from computer science to English literature. However, all our users share a common thread: you love working with data.

As our community grows, we want to make sure that Kaggle continues to be welcoming. To that end, we are introducing guidelines to ensure everyone has the same expectations about discourse in the forums:

Be patient. Be friendly.
Discuss ideas, don't make it personal.
Threats of any kind are unacceptable.
Low-level harassment is still harassment.
     
The Kaggle team determines whether content is appropriate. If you see something that violates these guidelines, you can bring it to our attention using the flag option on messages and topics. If you have a serious concern, you should report it to support@kaggle.com. All reports will be kept confidential.
'''


result = final_output.invoke(
    {"topic": topic}
)
print(result)


#Model graph structure:
'''
         +--------------------------------+
         | Parallel<notes,questions>Input |
         +--------------------------------+
                 **               **
              ***                   ***
            **                         **
+----------------+                +----------------+ 
| PromptTemplate |                | PromptTemplate | 
+----------------+                +----------------+ 
          *                               *
          *                               *
          *                               *
  +------------+                    +------------+   
  | ChatOllama |                    | ChatOllama |   
  +------------+                    +------------+   
          *                               *
          *                               *
          *                               *
+-----------------+              +-----------------+
| StrOutputParser |              | StrOutputParser |
+-----------------+              +-----------------+
                 **               **
                   ***         ***
                      **     **
        +---------------------------------+
        | Parallel<notes,questions>Output |
        +---------------------------------+
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