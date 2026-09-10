from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
    MessagesPlaceholder,
)


from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage,
)

from langchain.chat_models import init_chat_model

from langchain_core.output_parsers import StrOutputParser

from langchain_openai import ChatOpenAI

from dotenv import load_dotenv


load_dotenv()



parser = StrOutputParser()

prompt = ChatPromptTemplate.from_template("Write a short poem about {topic}")



llm = init_chat_model(
    model = "gpt-4o-mini",
    temperature = 0
)


chain = prompt | llm | parser

response = chain.invoke({"topic": "nature"})


print(type(response))



# ===== We have other output parser =====

# JsonOutputParser example
from langchain_core.output_parsers import JsonOutputParser


parser = JsonOutputParser()


prompt = ChatPromptTemplate.from_template("Return a JSON object with 'name' and 'age' for: {description}" )


chain = prompt | llm | parser

response = chain.invoke({"description": "a 25 developer named Moe"})

print(response)


# Recommended way
# PydanticOutputParser example
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

# Class model
class Person(BaseModel):
    name: str = Field(description="The person's name")
    age: int = Field(description="The person's age")


parser = PydanticOutputParser(pydantic_object=Person)

# Create class model --> Instantiate parser --> Create actual prompt

prompt = ChatPromptTemplate.from_template(
    "Return a JSON object with 'name', and 'age'for {description}:"
).partial(format_instruction = parser.get_format_instructions()) # return instuction format for JSON output


chain = prompt | llm | parser


result = chain.invoke({"description": "A 30-year-old artist named Maria"})
print(result)



# Recommended way for output parser
from pydantic import BaseModel, Field


#1. Create class
class MovieReview(BaseModel):
    title: str = Field(description="The title of the movie")
    review: str = Field(description="A brief review of the movie")
    rating: float = Field(description="The rating of the movie out of 10")


# Bind the schema to the model
llm = init_chat_model(
    model="gpt-4o-mini",
    temperature = 0
)
structured_model = llm.with_structured_output(MovieReview)

# model with structured output
result = structured_model.invoke("Review: Inception is a mind-bending thriller. 9/10")

print(result)
