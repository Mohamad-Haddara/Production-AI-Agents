"""
Prompt Templates and Messages in LangChain
"""


from pydoc import text
from langchain_core import messages
from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
    FewShotPromptTemplate,
    MessagesPlaceholder,
    few_shot,
)



from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage,
)


from langchain_openai import ChatOpenAI
from dotenv import load_dotenv




load_dotenv()


def demo_basic_templates():
    """
    Basic ChatpromptTemplate usage.
    """

    # Simple template
    simple_template = ChatPromptTemplate.from_template("Translate {text} to langauge {language}")

    messages = simple_template.format_messages(text="Hello, world!", language="French")
    
    print("Simple Template")
    print(f"{messages}")

    # Multi-message template
    multi = ChatPromptTemplate.from_messages(
        [
            ("system", "You are translator be concise"),
            ("human", "Translate '{text}' to {language}")

        ]
    )

    messages = multi.format_messages(text="Good morning", language = "Japanese")
    print("\nMulti-message template")

    for msg in messages:
        print(f"    {type(msg).__name__} : {msg.content}")


def demo_message_types():
    "Working with different message types"

    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # Build conversation manually
    messages = [
        SystemMessage(content="You are a math tutor. Be brief"),
        HumanMessage(content="What is 5 * 5?"),
        AIMessage(content="25"),
        HumanMessage("And If I add 10?")
    ]


    response = model.invoke(messages)

    print(f"Conversation result: {response.content}")



def demo_few_shot():
    """
    Few-shot prompting with examples.
    """

    # Define examples
    examples = [
        {"word":"happy", "opposite":"sad"},
        {"word":"fast", "opposite":"slow"},
        {"word":"big", "opposite": "small"}
    ]


    # Template for each example
    example_prompt = ChatPromptTemplate.from_messages(
        [
            ("human", "What is the opposite of '{word}'?"),
            ("ai", "The opposite of `{word}` is {opposite}.")
        ]
    )

    # Few-shot wrapper
    few_shot = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=examples
    )
    
    # Final prompt
    final_prompt = ChatPromptTemplate.from_messages(
        [
        ("system", "you give the opposite of words. Follow the examples"),
        few_shot,
        ("human", "What is the opposite of {word} ?")
        ]
    )


    # Test
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    chain = final_prompt | model

    response = chain.invoke({"word":"bright"})

    print(f"Few-shot result: {response.content}")



def demo_prompt_composition():
    """Compose prompts from reusable parts."""

    # Reusable system prompt
    persona = ChatPromptTemplate.from_messages(
        [("system", "You are a {role}. Your tone is {tone}.")]
    )


    # Reusable task prompt
    task = ChatPromptTemplate.from_messages([("human", "{task}")])


    # Combine
    full_prompt = persona + task

    # Test different combination
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    chain = full_prompt | model

    response = chain.invoke(
        {
            "role":"pirate captain",
            "tone": "adventurous",
            "task":"Tell me about your ship",
        }
    )


    response = chain.invoke(
        {
            "role": "scientist",
            "tone": "precise and academic",
            "task": "Explain photosynthesis"
        }
    )

    print(f"\nScientist: {response.content[:100]}...")




if __name__ == "__main__":
    print("="*50)
    print("Demo 1: Basic Templates")
    print("="*50)
    demo_basic_templates()


    print("\n" + "=" *50)
    print("Demo 2: Message Types")
    print("="*50)
    demo_message_types()


    print("\n" + "=" *50)
    print("Demo 3: Few Shot Examples")
    print("="*50)
    demo_few_shot()


    print("\n" + "=" *50)
    print("Demo 4: Compose Prompts")
    print("="*50)
    demo_prompt_composition()
    
