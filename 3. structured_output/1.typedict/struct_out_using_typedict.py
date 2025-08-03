from langchain_ollama import ChatOllama
from typing import TypedDict,Annotated,Optional,Literal
# TypedDict is a special class in Python's typing module that allows you to specify the expected types of dictionary keys and values.

model=ChatOllama(
    model="deepseek-r1:8b",
    temperature=0.5,
)
# simple typedict example
class Recipe(TypedDict):
    name: str
    ingredients: list[str]
    instructions: str

#annotated/optional typedict example, this allows you to add metadata to the fields so that ai can understand the context better
class Review(TypedDict):
    topic:Annotated[str, "The topic of the review"]
    rating: Annotated[int, "The rating out of 5"]
    summary: Annotated[str, "A brief summary of the review"]
    sentiment: Annotated[Literal["positive", "negative", "neutral"], "The sentiment of the review, e.g., positive, negative, neutral"]
    # Literal is used to specify that the sentiment can only be one of the specified values
    user: Annotated[Optional[str], "The user who submitted the review"]
    # Optional is used to indicate that the user field can be None

structured_output_1 = model.with_structured_output(Recipe)
structured_output_2 = model.with_structured_output(Review)

# It helps with type checking and code clarity when working with dictionaries that have a fixed structure.
result = structured_output_1.invoke('''Recipe for a simple pasta dish:''')
review = structured_output_2.invoke('''Review for the pasta dish: I loved it! The flavors were amazing and the texture was perfect. I would give it a 5 out of 5.''')
print(result)
print(review)