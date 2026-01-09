# pip install -qU langchain "langchain[anthropic]"
from langchain.agents import create_agent
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain.messages import HumanMessage,AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()

def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"
 

# Setup model and agent
model = ChatOllama(model="granite4:1b")

model_google = ChatGoogleGenerativeAI(
    model="gemini-3-flash-preview",
    temperature=1.0,  # Gemini 3.0+ defaults to 1.0
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)

agent = create_agent(
    model=model,
    tools=[get_weather],
    system_prompt="You are a helpful assistant",
)

user = [HumanMessage(content="What's the weather like in New York?")]

# LangGraph agents expect dict input with "messages" key
output = agent.invoke({"messages": user})

# Extract the final AI message from the output
messages = output["messages"]

final_ai_message = next(
    msg for msg in reversed(messages) if isinstance(msg, AIMessage)
)

print(final_ai_message.content)
