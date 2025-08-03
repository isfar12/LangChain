from pydantic import BaseModel, EmailStr, Field
from typing import Optional, Annotated

# Pydantic is a data validation and settings management library for Python, which uses Python type annotations.
class User(BaseModel):
    name: str="unknown" # you can set a default value
    email : EmailStr # automatically validates email format
    age: Optional[int] = Field(None, ge=0, description="The age of the user")  # age must be a non-negative integer, can be None

    
# using Annotated to add metadata
class UserNew(BaseModel):
    name: Annotated[str, Field(default="unknown", description="Name of the user")]
    email: EmailStr
    age: Optional[int] = Field(None, ge=0, description="The age of the user, must be non-negative")


user_data = User(name="John Doe", email="john.doe@example.com", age=30)
print(dict(user_data))