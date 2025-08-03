from langchain_ollama import ChatOllama
from typing import List, Literal
from pydantic import BaseModel, Field
import json

model = ChatOllama(
    model="llama3.1",
    temperature=0.1,  # Lower temperature for more consistent JSON output
)

# Define the schema as a dictionary instead of JSON string
json_schema = {
    "title": "ProductReview",
    "description": "Schema for summarizing a product review",
    "type": "object",
    "properties": {
        "title": {
            "type": "string",
            "description": "A short title summarizing the review"
        },
        "summary": {
            "type": "string", 
            "description": "A concise summary of the review content"
        },
        "sentiment": {
            "type": "string",
            "enum": ["positive", "neutral", "negative"],
            "description": "Overall sentiment of the review"
        },
        "keywords": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Important keywords extracted from the review"
        },
        "rating": {
            "type": "number",
            "minimum": 1,
            "maximum": 5,
            "description": "Star rating from 1 to 5"
        }
    },
    "required": ["title", "summary", "sentiment"],
    "additionalProperties": False
}

structured_output_2 = model.with_structured_output(json_schema)

# Test the structured output
user = structured_output_2.invoke(
    """
Summarize this product review according to the JSON schema.

The iPhone 15 Pro Max is fantastic! The titanium frame feels premium, the camera takes stunning photos even in low light, and the battery lasts all day. It's a bit pricey, but absolutely worth it for tech lovers.

Return ONLY a JSON object with:
- title (short heading)
- summary (2–3 sentences)
- sentiment (positive, neutral, or negative)
- keywords (list of important words)
- rating (1 to 5 stars)
"""
)

print("Structured JSON Output:")
print(json.dumps(user, indent=2))
print(f"\nTitle: {user['title']}")
print(f"Summary: {user['summary']}")
print(f"Sentiment: {user['sentiment']}")
print(f"Keywords: {user['keywords']}")
print(f"Rating: {user['rating']}")