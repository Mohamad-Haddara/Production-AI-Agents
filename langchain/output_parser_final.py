"""
Output Parsers and Structured Output in LangChain
"""

from locale import strcoll
from langchain_core.output_parsers import (
    StrOutputParser,
    JsonOutputParser,
    PydanticOutputParser
)


from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import List, Optional
from dotenv import load_dotenv


load_dotenv()


model = ChatOpenAI(model = "gpt-4o-mini", temperature=0)

def demo_str_parser():
    """Basic string output parser."""

    prompt = ChatPromptTemplate.from_template(
        "Give me one-word answer: What color is the sky?"
    )

    parser = StrOutputParser()

    chain = prompt | model | parser

    result = chain.invoke({})

    print(f"Result: {result} (type: {type(result).__name__})")


def demo_json_parser():
    """JSON output parser"""

    prompt = ChatPromptTemplate.from_template(
        "Return a JSON object with keys 'city' and 'country' for: {place}\n"
        "Return ONLY valid JSON, explanation."
    )

    parser = JsonOutputParser()

    chain = prompt | model | parser

    result = chain.invoke({"place": "The Effiel Tower"})

    print(f"Result: {result}")
    # Extract values from dictionary keys
    print(f"City: {result['city']}, Country: {result['country']}")



def demo_pydantic_parser():
    """Pydantic output parser for type-safe structured data."""


    # Define Schema 
    class Recipe(BaseModel):
        name: str = Field(description="Name of the recipe")
        ingredients: str = Field(description="List of ingredients")
        prep_time_minutes: int = Field(description="Preperation time in minutes")
        difficulty: str = Field(description="easy, medium, or hard")

    
    parser = PydanticOutputParser(pydantic_object=Recipe)

    prompt = ChatPromptTemplate.from_template(
        "Create a simple recipe for: {dish}\n\n{format_instructions}"
    ).partial(format_instruction=parser.get_format_instructions())


    chain = prompt | model | parser


    result = chain.invoke({"dish": "scrambled eggs"})
    print(f"Recipe: {result.name}")
    print(f"Ingredients: {result.ingredients}")
    print(f"Prep time: {result.prep_time_minutes} mins")



def demo_stuctured_output():
    """Modern with_structured_output() method"""

    class TaskExtraction(BaseModel):
        """Extracted task information."""
        
        task: str = Field(description="The main taks to do")
        priority: str = Field(description="high, medium, or low")
        deadline: Optional[str] = Field(description="Deadline if mentioned")
        assign: Optional[str] = Field(description="Person assigned of mentioned")


    
    # Bind schema to model
    structured_model = model.with_structured_output(TaskExtraction)

    # No need for format instructions - it is automatic
    prompt = ChatPromptTemplate.from_template("Extract task information from : {text}")

    chain = prompt | structured_model

    texts = [
        "John needs to finish the report by Friday - it is urgent",
        "We should update the docs sometime next week",
        "Critical: Fix the login bug ASAP",
    ]


    print("Task Extractions:")
    for text in texts:
        result = chain.invoke({"text": text})
        print(f"\n Input: {text}")
        print(f" Task: {result.task}")
        print(f" Priority: {result.priority}")
        print(f" Deadline: {result.deadline}")

#V.I - complex real-world application
def demo_complex_schema():
    """Complex nested schema with structured output."""

    class Address(BaseModel):
        street: str 
        city: str
        country: str

    class Company(BaseModel):
        name: str
        industry: str
        employee_count: int
        headquarter: Address
        products: List[str]


    structured_model = model.with_structured_output(Company)


    prompt = ChatPromptTemplate.from_template(
        "Extract company information from: {text}"
    )

    chain = prompt | structured_model


    result = chain.invoke(
        {
            "text":"Apple Inc. is a tech company with 160,000 employees based in"
            "Cupertino, California, USA. They make iPhones, MacBooks, and iPads."
        }
    )


    print(f"Company: {result.name}")
    print(f"HQ: {result.headquarters.city}, {result.headquarters.country}")
    print(f"Products: {result.products}")


# Extract data from the text

