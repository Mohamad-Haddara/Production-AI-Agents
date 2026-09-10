from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser

"""Prompt Templates and Messages in LangChain"""
load_dotenv()


# chatprompttemplate
prompt = ChatPromptTemplate.from_template("Tell me a {adjective} joke about {topic}.")


# format and inspect
messages = prompt.format_messages(adjective="funny", topic="chickens") #Formatted Message -> HumanMessage

print(messages)


# Multi-message templates -- Pass a list of messages
prompt = ChatPromptTemplate.from_template(
    [
        #SystemMessage(content="") or
        (
            "system", "You are a helpful assistant that translate {input_language} to {output_language}."
        ),
        (
            "human", "Translate the following text: {text}"
        )
    ]
)

# I can use prompt to fill the input variables. Format messages in prompt using format_messages methods
messages = prompt.format_messages(
    input_language="English",
    output_language="French",
    text = "I love programming"
)


print(messages)

model = init_chat_model(model="gpt-4o-mini", temperature=0)
response = model.invoke(messages)
print(response.content)


# Message Types
from langchain_core.messages import(
    AIMessage,
    SystemMessage,
    HumanMessage,
    ChatMessage,
    ToolMessage
)


messages = [
    HumanMessage(),
    ToolMessage(),
    SystemMessage(),
    ChatMessage(),
    AIMessage()
]


# Fewshot example
from langchain_core.prompts import FewShotChatMessagePromptTemplate

# create list of examples
examples [
    {},
    {}
]


example_prompt = ChatPromptTemplate.from_messages([

])