from typing import TypedDict
# TypedDict is a special class in Python's typing module that allows you to specify the expected types of dictionary keys and values.
# It helps with type checking and code clarity when working with dictionaries that have a fixed structure.
class Person(TypedDict):
    name: str
    age: int
    is_student: bool

new_person: Person = {
    "name": "Alice",
    "age": 30,
    "is_student": False
}
print(new_person)