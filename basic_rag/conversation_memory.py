"""
Conversation Memory in LangChain.
Modern approaches to maintaining conversation context
"""
# How memory work in conversation - save the context of the conversation as it goes

from langchain_openai import ChatOpenAI
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder # Placeholder for history
from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
    AIMessage,
    content,
    trim_messages
)


from langchain_core.chat_history import (
    InMemoryChatMessageHistory,
    BaseChatMessageHistory,
)


from langchain_core.runnables.history import RunnableWithMessageHistory # manage chat message history
from langchain_core.output_parsers import StrOutputParser
from typing import Dict
from dotenv import load_dotenv
from numpy import isin
from numpy.f2py.cfuncs import userincludes
from openai.types.chat import chat_completion_named_tool_choice_custom_param


load_dotenv()


def basic_memory():
    """
    Basic conversation memroy with RunnableWithMessageHistory
    """

    print("="*60)
    print("BASIC CONVERSATION MEMORY")
    print("Using RunnableWithMessageHistory (modern approach)")
    print("="*60)


    llm = init_chat_model(model="gpt-4o-mini")

    # Prompt with history placeholder
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are helpful assistant. Be concise"),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ])


    chain = prompt | llm | StrOutputParser()


    # Session Storage
    store: Dict[str, InMemoryChatMessageHistory] = {}


    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in store:
            store[session_id] = InMemoryChatMessageHistory()
        
        return store[session_id] # return memory chat message corresponding this session id

    
    # Wrap with history
    chain_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history"
    )

    
    # Configuration for this session ~ In production I would have a system that would create a dynamic user or session ID
    config = {"configurable": {"session_id":"user_123"}}

    # Actual Conversation Messages
    messages = [
        "Hi! My name is Moe",
        "I am learning about LangChain.",
        "What is my name and what am I learning?"
    ]



    print("\nConversation:")
    for msg in messages:
        print(f"\nUser: {msg}")
        response = chain_with_history.invoke({"input": msg}, config=config)
        print(f"AI: {response}")


    # Show stored history
    print(f"--- Stored History ({len(store["user_123"].messages)} messages) ---")
    for msg in store['user_123'].messages:
        role = "Human" if isinstance(msg, HumanMessage) else "AI"
        print(f"{role}:{msg.content[:50]}...")



# Multiple Sessions Memory - Allow us to handle multiple conversation sessions

def multi_sessions():
    """
    Save conversaiton in session which will have id, so we can add and retain it
    """

    print("="*60)
    print("MULTIPLE CONVERSAION SESSIONS")
    print("Each user gets their own memory")
    print("="*60)


    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Remember user details"),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ])

    llm = init_chat_model(model="gpt-4o-mini", temperature = 0.3)


    chain = prompt | llm| StrOutputParser()


    store: [Dict, InMemoryChatMessageHistory] = {}
    

    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in store:
            store[session_id] = InMemoryChatMessageHistory()

        return store[session_id]
    

    chat_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )


    # Simulate two different users
    user_a_config = {"configurable": {"session_id":"user_a"}}
    user_b_config = {"configurable": {"session_id":"user_b"}}


    # User A conversation
    print("\n--- User A ---")
    print("User A: My favorite language is Python")
    resp = chat_with_history.invoke({"input": "My favorite language is Python"}, config=user_a_config)
    print(f"AI: {resp}")


    # User B conversation
    print("\n--- User B ---")
    print("User B: I love JavaScript")
    resp = chat_with_history.invoke({"input":"I love JavaScript"}, config=user_b_config)
    print(f"AI: {resp}")


    # Ask each user about their preference
    print("\n--- Asking each about their preference ---")
    print("\nUser A: What's my favorite language?")
    resp = chat_with_history.invoke({"input": "What is my favorite language?"}, config=user_a_config)
    print(f"AI: {resp}")

    
    print("\n--- Asking each about their preference ---")
    print("\nUser B: What's my favorite language?")
    resp = chat_with_history.invoke({"input": "What is my favorite language?"}, config=user_b_config)
    print(f"AI: {resp}")



# Message Trimming - Trim messages to fit the context window (V.I skill to have)
def message_trimming():
    """
    Trim messages to fit context window.

    Imagine that I have huge conversation in database for AI memory. Trimming is very helpful to trim tokens and fit context window
    
    """
    
    print("="*60)
    print("MESSAGE TRIMMING")
    print("Keep conversation within token limits")
    print("="*60)

    llm = init_chat_model(model="gpt-4o-mini")


    # Simulate a long coversation (like memory history)
    messages = [
        SystemMessage(content="You are helpful coding assistant."),
        HumanMessage(content="What is Python?"),
        AIMessage(content="Python is a high-level programming language known for"),
        HumanMessage(content = "How do I install it?"),
        AIMessage(content="I can install Python from python.org or use package"),
        HumanMessage(content="What is pip?"),
        AIMessage(content="Pip is a Python package installer."),
        HumanMessage(content="Can you summarize everything we discussed?")
    ]


    # To measure trimming, we must check the length of the message
    print(f"\nOriginal: {len(messages)} messages")


    # Trim to last N tokens - set strategy for trimming
    trimmed = trim_messages(
        messages, # list of the message conversation history
        max_tokens=60,
        strategy="last", # we have first also
        token_counter= llm,
        include_system=True, # Always keep system message
        allow_partial=False,
    )


    print(f"After trimming (max 60 tokens): {len(trimmed)} messages")
    print("\nTrimmed messages:")
    for msg in trimmed:
        role = type(msg).__name__.replace("Message", "")
        print(f"    {role}: {msg.content[:60]}...")



# Windowed Memory - Every message send to the LLM will cost tokens, so in the long converssation - sending the entire history every single time. Hit model context window limit
# Sliding window -> only keep the last K exchanges - Older ones get dropped
def windowed_memory():
    """
    Implement sliding window memory manually.
    """ 

    llm = init_chat_model(model="gpt-4o-mini")

    print("="*60)
    print("WINDOWED MEMORY (Keep Last K)")
    print("Fixed-size conversation window")
    print("="*60)



    # Create a internal class called window chat history - Extending the langchain built in memory chat message history - key is to add messages method
    class WindowChatHistory(InMemoryChatMessageHistory):
        """Chat history that keeps only last k message pairs."""

        def __init__(self, k:int =3):
            super().__init__()
            self.k = k

        # It ensure that every time a new message is added by human or AI -> it is gonna check -> do I have more than K times 2 messages? Note: 1 exchange = 1 human message + AI response. So 
        # Every message I send, I will have 2 messages back
        def add_messages(self, messages):
            super().add_messages(messages)
            # Keep only last k pairs (2k messages: human + AI)
            if len(self.messages) > self.k * 2:
                self.messages = self.messages[-(self.k*2):] # slice in the end - keep only the last k2 messages


    store: Dict[str, WindowChatHistory] = {}



    def get_session_history(session_id):

        if session_id not in store:
            store[session_id] = WindowChatHistory(k=2)

        store[session_id]

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are helpful assistant."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ])


    chain = prompt | llm | StrOutputParser




    chain_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history"
    )


    config = {"configurable": {"session_id": "windowed_test"}}

    # Simulate a conversation with more than 2 pairs
    exchange = [
        "My name is Moe",
        "I live in doha",
        "I work as AI Engineer",
        "I have 2 cats",
        "What do you remember about me?",
    ]


    print("\n Conversation with k=2 window:")
    for i, msg in enumerate(exchange, 1):
        print(f"\nUser: {msg}")
        response = chain_with_history.invoke({"input":msg}, config=config)
        print(f"AI: {response}")


        # Show window state after each exchange so students SEE it sliding
        history = store["windowed_test"].messages
        print(f"    [Window: {len(history)} msgs]", end="")
        facts_in_memory = [
        m.content[:40]    for m in history if isinstance(m, HumanMessage)
        ]

        print(f"Remembers: {facts_in_memory}")


    # Final state - show what survived and what was lost
    print("\n" + "=" *60)
    print("RESULT: Window only kept last 2 exchanges!")
    print("Lost: name (Moe), city(Doha), AND job (AI Engineer)")
    print("This is the tradeoff: fixed memory = predictable cost, but older context is lost")



# Summary Memory
def summary_memory():
    """
    Implementing conversation summarization.
    End-to-end summary memory: auto-summarize old messages, keep recent ones for downstream

    we are not deleting anything, but just compressing things (best in both world)
    Look at this strategy to maximize our RAG system - think how to make application save memory in conversation, but as our conversation is getting long, we need to think of different
    strategies to make sure that we don't go over context window
    """

    print("="*60)
    print("SUMMARY MEMORY")
    print("Summarize older messages to save tokens")
    print("="*60)

    # --- Setup ---
    summary_llm = ChatOpenAI(model = "gpt-4o-mini", temperature=0) # Deterministic for summary
    chat_llm = ChatOpenAI(model="gpt-4o-mini", temperature = 0.7) # Creative



    # The conversation prompt: summary of old context + recent messages
    chat_prompt = ChatPromptTemplate.from_messages(
        [
            (
            "system", 
            "You are helpful assistant. Be Concise.\n\n"
            "Summary of earlier conversation:\n {summary}"
            ),
            MessagesPlaceholder(variable_name="recent_messages"),
            ("human", "{input}"),

        ]
    )


    chat_chain = chat_prompt | chat_llm | StrOutputParser()

    # The summarization prompt: Compress messages into a running summary
    summarize_prompt = ChatPromptTemplate.from_template(
    
    "Condense the current summary and new messages into a single updated summary."
    "(2-3 sentence) Preserve all key facts about the user.\n\n"
    "Current Summary:\n{current_summary}\n"
    "New messages:\n{new_messages}\n\n"
    "Updated summary:"
    
    )

    summarize_chain = summarize_prompt | summary_llm | StrOutputParser()

    # --- State ---
    running_summary = "" # starts empty
    recent_messages = [] # full message objects
    MAX_RECENT = 4 # keep last 4 messages (2 exchanges) before summarizing


    # --- Conversation ---
    exchange = [
    "My name is Moe and I live in doha",
    "I work as AI engineer buidling RAG system",
    "I have 2 cats and j and r",
    "What do you remember about me? list everything",
    ]

    print(f"\nConfig: keep last {MAX_RECENT} messages, summarize the rest\n")

    for user_input in exchange:

        print(f"User: {user_input}")

        # 1. Call the LLM with summary + recent messages + new input
        response = chat_chain.invoke(
            {
                "summary": (
                   running_summary if running_summary else "No prior conversation."
                ),
                "recent_messages": recent_messages,
                "input": user_input
            }
        )
    
        print(f"AI: {response}")

        # 2. Add this exchange to recent messages
        recent_messages.append(HumanMessage(content=user_input))
        recent_messages.append(AIMessage(content=response))

        # 3. If recent messages exceed limit, summarize the oldest ones (if exceed max_recent it will trigger summarization)
        if len(recent_messages) > MAX_RECENT:
            # Take the oldest messages that will be summarized away
            messages_to_summarize = recent_messages[:-MAX_RECENT]
            formatted = "\n".join(
                f"{'Human' if isinstance(m, HumanMessage) else 'AI'}: {m.content}"
                for m in messages_to_summarize
            )

            # Update the running summary
            running_summary = summarize_chain.invoke(
                {
                    "current_summary":(
                        running_summary if running_summary else "None yet." 
                    ),
                    "new_messages":formatted,
                }
            )


        # Keep only the most recent messages
        recent_messages = recent_messages[-MAX_RECENT:]

        #print(    f"  >>> Summarized! Compressed {len(messages_to_summarize)} old messages")

        print(f"    >>> Summary: {running_summary}")
        print(f"    >>> Recent buffer: {len(recent_messages)} messages")

    print()

    # --- Final state ---
    print("="*60)
    print("FINAL MEMORY STATE")
    print("="*60)
    print(f"\nRunning summary (compressed old context):\n {running_summary}")
    print(f"\nRecent messages kept verbatim ({len(recent_messages)}):")
    
    for msg in recent_messages:
        role = "Human" if isinstance(msg, HumanMessage) else "AI"
        print(f"    {role}: {msg.content[:80]}")
    
    print(f"\nKey insight: ALL facts preserved (name, city, job, cats, course)")
    print(f"But token cost stays bounded -- old messages are compressed, not deleted")

   





if __name__ == "__main__":
    #basic_memory()
    #multi_sessions()
    #message_trimming()
    #windowed_memory()
    summary_memory()