from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline


generator=pipeline(
    model="./models/falcon-1b",
    task="text-generation",
    max_length=100,
    temperature=0.5,
    top_p=0.9,
    do_sample=True,
)

llm=HuggingFacePipeline(pipeline=generator)
result=llm.invoke("What is the capital of Bangladesh?")
print(result)