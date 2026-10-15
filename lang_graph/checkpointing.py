"""
Without checkpointing, every app that involve call is stateless
which means it doesn't have any state, it is not saving anything.
So the graph runs, returns a result and then forgets everything.

Create 3 problems:
1) No conversation memory
2) If something happens - crashes kill everything - if something happens we can go to previous state
3) No human in the loop - I can't pause a graph (like I can let a human review something and then resume later)
without persistance --> the moment execution stops, the state is gone forever.

Solution: is checkpointing ~ like a video game save system - after every node executes, the langgraph saves a snapshot of the entire state - Each snapshot is a checkpoint
"""

import enum
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver # save memory in database
from typing_extensions import TypedDict, Annotated
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
import operator
import tempfile
from dotenv import load_dotenv

load_dotenv()


llm = ChatOpenAI(model="gpt-4o-mini", temperature = 0.0)

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add] # add message as a reducer - BaseMessage will be all input and output of the chat


def memory_saver():
    """In-memory checkpointing for development."""

    def chat(state: TypedDict) -> dict:
        # send the message to the llm and append the response
        response = llm.invoke(state['messages'])
        return {
            "messages": [response]
        }


    graph = StateGraph(ChatState)

    graph.add_node("chat", chat)
    
    graph.add_edge(START, "chat")
    graph.add_edge("chat", END)


    # In memory checkpointer
    saver = MemorySaver()

    app = graph.compile(checkpointer=saver)

    # Configure with thread
    config = {"configurable": {"thread_id":"user-123"}}
    
    print("Memory Saver (Multi-turn conversation):\n")

    # Turn 1
    result = app.invoke(
        {"messages": [HumanMessage(content="My name is Moe")]},
        config
    )

    # print the result comes out from LLM
    print(f"Turn 1 - AI: {result['messages'][-1].content}")


    # Turn 2 - Conversation continues - testing if the checkpoint saved the information and then return the result
    result = app.invoke(
        {"messages": [HumanMessage(content="What is my name?")]},
        config
    )

    print(f"Turn 2 - AI: {result['messages'][-1].content}")


    # Check full history
    state = app.get_state(config)
    print(f"\nTotal messages in state: {len(state.values['messages'])}") # look at all the values of the messages in the list
    
    # we are going to look at all of the messages field under the state


# Now we persist locally on SQLite database ~ durable storage
def sqlite_persistance():
    """
    SQLite persistance for durable storage.
    
    This is the basis of production systems - Because we can persist the information for durable storage and use SQLite persistance for durable storage
    by invoking the SQLite persistant.

    Instead of svaing in memory (RAM), we save it in database.

    When we run it:
        - session 1 saved the checkpoints to the SQLite file on disk
        - session 2 open the same sqlite file, used the same thread ID. Even if the crash happens, nothing will be lost.
        We use thread_id to go and find the correct history conversation ~ which include the secret codes.

        So the only changes between here and previous code is the memory saver.
        We now have the SQLite saver from from_conn_string and pass db_path.

        In production I will use PostgreSaver with a real database  - and we follow the same pattern
    """

    # First node
    def chat(state: ChatState) -> dict:
        response = llm.invoke(state['messages'])
        return {'messages': [response]}

    # Setup Graph
    graph = StateGraph(ChatState)
    graph.add_node("chat", chat)
    
    graph.add_edge(START, "chat")
    graph.add_edge("chat", END)


    # Create temp database
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    print(f"\nSQLite Persistance:")
    print(f"Database: {db_path}\n")


    # First session
    # In production I will use PostgreSaver with a real database
    with SqliteSaver.from_conn_string(db_path) as saver:
        app = graph.compile(checkpointer=saver)
        config = {"configurable": {"thread_id":"persistent-user"}}

        result = app.invoke(
            {"messages": [HumanMessage(content="Remember: The secret code is ALPHA-7")]},
            config
        )

        print(f"Session 1 - Stored secret code")

    # Simulate app restart - new session
    # Second session - Retrieve stored secret code
    with SqliteSaver.from_conn_string(db_path) as saver:
        app = graph.compile(checkpointer=saver)
        config = {"configurable": {"thread_id":"persistent-user"}}

        result = app.invoke(
            {"messages": [HumanMessage(content="What was the secret code?")]},
            config
        )

        print(f"Session 2 - AI {result['messages'][-1].content}") # it will answer the secret because it is saved in the database 


def state_inspection():
    # Inspect and manipulate checkpoint state

    def chat(state: ChatState) -> dict: 
        response = llm.invoke(state['messages'])
        return {
            "messages": [response]
        }


    graph = StateGraph(ChatState)

    graph.add_node("chat", chat)
    graph.add_edge(START, "chat")
    graph.add_edge("chat", END)


    # In memory saver
    memory = MemorySaver()

    app = graph.compile(checkpointer=memory)
    config = {"configurable": {"thread_id":"inspect-demo"}}

    # Build up some state
    app.invoke({"messages": [HumanMessage(content="Hello")]}, config)
    app.invoke({"messages": [HumanMessage(content="How are you?")]}, config)


    # Get the current state
    state = app.get_state(config)


    print("Current State:")
    print(f"    Next node: {state.next}")
    print(f"    Message count: {len(state.values['messages'])}")


    # Get state history - 
    # This matter for:
    # 1) debugging - I can see how state look like for every step V.I
    # 2)Rewinding - I can go back to a previous checkpoint if something went wrong
    # 3) Auditing - Biggest reason because we can have full history of everything happened in a conversation
    print("\nState history:")
    for i, snapshot in enumerate(app.get_state_history(config)): # this give us a every checkpoint ever saved for this threat
        print(f"    Checkpoint {i}: {len(snapshot.values['messages'])} messages")
        if i>=3:
            print(" ...")
            break

# Most powerful patterns - like git branches for conversations
def branching_conversation():
    # Branching conversation from checkpoint

    def chat(state: ChatState):

        response = llm.invoke(state['messages'])
        return {
            "messages": [response]
        }

    graph = StateGraph(ChatState)
    graph.add_node("chat", chat)
    graph.add_edge(START, "chat")
    graph.add_edge("chat", END)


    memory = MemorySaver()
    app = graph.compile(checkpointer=memory)


    print(f"Branching Conversation:")

    # Main Conversation
    main_config = {"configurable": {"thread_id": "main"}}
    app.invoke({"messages": [HumanMessage(content="What's the weather like?")]}, main_config)

    # Get checkpoint to branch from - we get the main config to get that main state (main_state ~ snapshot)
    main_state = app.get_state(main_config)

    # Branch A ~ Beach vacation
    branch_a_config = {"configurable": {"thread_id": "branch-beach"}}
    
    # Copy state to a new thread
    app.update_state(branch_a_config, main_state.values)

    # we continue on branch independently
    result_a = app.invoke(
        {"messages": [HumanMessage(content="What about a beach vacation?")]},
        branch_a_config
    )

    print(f"Branch A (Beach): {result_a['messages'][-1].content[:100]}...")

    # Branch B -  Mountain Adventure
    branch_b_config = {"configurable": {"thread_id" : "branch_mountain"}}
    app.update_state(branch_b_config, main_state.values) 

    result_b = app.invoke(
        {"messages": [HumanMessage(content="What about mountain hiking?")]},
        branch_b_config
    )

    print(f"Branch B (Mountain): {result_b['messages'][-1].content[:100]}...")



    # All 3 threads are independent, so changes wont affect the others - This is a very fundamental way of thinking about this 
    # In real-world I will have different use cases - let say I want to do A/B testing - SO I have different agent strategies on the same conversation - I can use strategy ~ this branching conversations
    # Use What if ~ I try 2 approaches and keep the better one - So I can do it easily by using branching



def checkpoint_internals():
    """
    Peek inside a checkpoint - see exactly what LangGraph saves.

    Uses 2-node graph so we generate multiple checkpoints,
    then walks through every field in the checkpoint object.

    To see how things are saved inside the states and all of that
    """

    # --- Build a 2-node graph so we get several checkpoints ---
    class TaskState(TypedDict):
        messages: Annotated[list[BaseMessage], operator.add] # reducer: concatenate all the messages coming in, instead of deleting 
        step: str

    
    def analyze(state: TaskState) -> dict:
        response = llm.invoke(state['messages'])
        return {'messages': [response]}

    def summarize(state: TaskState) -> dict:
        summary_prompt = [
            HumanMessage(
                content=f"Summarize this in one sentence: {state['messages'][-1].content}"
            )
        ]

        response = llm.invoke(summary_prompt)
        return {"messages": [response], "step":"summarized"}

    
    graph = StateGraph(TaskState)
    graph.add_node("analyze", analyze)
    graph.add_node("summarize", summarize)
    graph.add_edge(START, "analyze")
    graph.add_edge("analyze", "summarize")
    graph.add_edge("summarize", END)

    memory = MemorySaver()
    app = graph.compile(checkpointer=memory)
    config = {"configurable": {"thread_id":"internal-demos"}}

    print("\nCheckpoint Internals")
    print("="*55)
    print("Graph: START -> [analyze] -> [summarize] -> END")
    print("="*55)

    # Run the graph
    app.invoke(
        {"messages": [HumanMessage(content="Explain why the sky is blue")]},
        config
    )

    # ==================================================================
    # PART 1: What os he current state snapshot?
    # ==================================================================
    print("\n--- PART 1: Current State Snapshot (app.get_state) ---\n")

    state = app.get_state(config)

    # state.values - our actual TypedDict data
    print("1) state.values (our state data)")
    print(f"    step: {state.values['step']}")
    print(f"    messages: {len(state.values['messages'])} total")

    for i, msg in enumerate(state.values['messages']):
        role = "Human" if isinstance(msg, HumanMessage) else "AI"
        print(f"    [{i}] {role}: {msg.content[:80]}...")


    # state.next - which node runs next (empty = graph finished)
    print(f"\n2) state.next (pending node): ")
    print(f"    {state.next if state.next else '() - graph finished, no pending node'}")

    # state.config - the config that produced this snapshot
    print("\n3) state.config (thread + checkpoint IDs):")
    print(f"    thread_id:  {state.config['configurable']['thread_id']}")
    print(f"    checkpoint_id: {state.config['configurable']['checkpoint_id']}")

    # state.metadata - who created this checkpoint
    print("\n4) state.metadata (provenance info):")
    print(f"    source: {state.metadata.get('source','N/A')}")
    print(f"    source: {state.metadata.get('step','N/A')}")
    print(f"    source: {state.metadata.get('writes','N/A')}")


    # state.parent_config - pointer to the PREVIOUS checkpoint
    print(f"\n5) state.parent_config (previous checkpoint):")
    if state.parent_config:
        print(
            f"  parent checkpoint_id: {state.parent_config['configurable']['checkpoint_id']}"
        )

    else:
        print(f"    None - this is the very first checkpoint")

    # state.created_at - timestamp
    print("\n6) state.created_at (when saved):")
    print(f"    {state.created_at}")

    
    
    # ==================================================================
    # PART 2: Walk through ALL checkpoints (time travel)
    # ==================================================================

    print("\n=== PART 2: Full Checkpoint History (app.get_state_history) ===\n")
    print("LangGraph saves a checkpoint at EACH step. Let's see them all:\n")

    for i, snapshot in enumerate(app.get_state_history(config)):
        step_num = snapshot.metadata.get("step", "?")
        source = snapshot.metadata.get("source", "?")
        writes = snapshot.metadata.get("writes", {})
        msg_content = len(snapshot.metadata.get("messages", []))
        checkpoint_id = snapshot.config["configurable"]["checkpoint_id"]
        current_step = snapshot.values.get("step", "")




    # Which node just wrote to this checkpoint?
    node_name = list(writes.keys())[0] if writes else "-"

    print(f" Checkpoint {i}:")
    print(f"    id:     {checkpoint_id[:30]}...")
    print(f"    source:     {source}")
    print(f"    step:       {step_num}")
    print(f"    written by:     {node_name}")
    print(f"    state.step:     {current_step}")














if __name__ == "__main__":
    #memory_saver()
    #sqlite_persistance()
    #state_inspection()
    #branching_conversation()
    checkpoint_internals()