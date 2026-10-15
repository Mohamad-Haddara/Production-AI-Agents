"""
LangChain is framework for bulding generative AI applications (like: RAG).
LCEL
Open-source framework
Limitation: bulding complex agentic systems
With LCEL we can't make cyckes - we can only have the acyclic graphs

ReAct algo there is a while loop in the source code
Here come LangGraph ~ State Machine -> Decide output step - Decide which steps to take (cycles)

With LangGraph we can implement cycles - this feature is very important when I want to build very
complex agent.

We define flow of our program and we can blend in with LangGraph the LLM, and LLM help us
where do we want to go in the flow ~ Flow A or Flow B or Finsh

LangGraph - Building agents as graphs - the entire flow of the agent can be expressed as a graph,
as a graph with cycles.

With LangGraph, we can build advanced systems



LangChain Vs LangGraph
Building AI system

LLM router is allow LLM to decide where do we want to go. We use LLM to decide whether
we are going to execute code in Branch 1 (search in database) or Branch 2 (search in web).
We are using reasoning power of an LLM ~ decide which step to take ~ Agent means there is 
cycle to decide which steps to take 

Agent is a control flow that llm decides where to go

Chain = is one directional - move from the left to the right
Agent = We have cycles -> which will give agentic properties. They use function calling to decide which
steps to take.

function calling -> we can send a description of tools, and those function we can execute in
our backend. WE send to the llm the description of those functions, their arguments, their
title and what their return value is. We do this with tool decorator.

Then LLM invoke these functions with those argument and get back the response.


ReAct
LLM decide whether we need to use a tool (call to an API or database) - if we need to call the
tool the llm call this tool with the arguments that the LLM chose to execute this tool.
We get an answer and feedback everything back to the LLM which decide to use another tool
or return the answer to the user.



Chain -> Not flexible and more reliable
Agent -> Flexible and not reliable

State Machine --> LangGraph -> we represent our software as a graph with nodes and edges
we represent it as a state machine which can have cycles, but we can control the flow as 
developers, and LLM decide which flow to go..

With LangGraph we architect our software as state machine (nodes, and edges) and displayed
as graph -> building agentic application

In LangGrpah
As developer -> we define the control flow (reliable) ~ like step 1 and step 2 and END
LLM make decisions inside the flow and what to execute in this flow and with cycles
We represnt this flow with nodes and with edges.

LLM can use conditional branching to decide where to go and execute node 1 and node 2


Start -> Node 1 -> Edge (Planning: Edges can route control flow based on LLM decisions) -> Node 2 (Tools: Nodes can call tools and modify state) -> End


LangGraph = Express control flows as graphs


Two important concept for Graph
1) Data structure -> represent relationships -> consists of nodes ~ also as known as vertices/edges that connect those nodes ~ used in social networks - transportation map
2) State Machine -> model of computation where it consists of states and transition between those states
State are nodes and the transitions are edges

LangGraph is built on top of langchain, and with LangGraph we can describe our flows with node and edges
to build complex agent application


Flow Engineering



LangGraph Core Components:
-------------------------------
With langgraph I am going to implement control flow or state machine or graph

1) Nodes
Python functions:
    - Code that calls LLM
    - Web search
    - LLM agent 

Node are special function that always recieves as an input - the current graph state,
which is going to have all the information that the nodes needs in order to do its work.

The return function is updating the state, and it is always a dictionary
he keys what we want to update in our state.

In LangGraph every node is going to update the state, and edges or conditonal edges
will rely on state whether to go to Node A or Node B.





2) Edges
connect these node
    - add_edge
    - add_conditional_edges ~ dynamic


No Operation nodes:
START node -> The entry point for our graph execution.
END node -> last node to be executed

State (V.I concept)
A state is a dictionary that is going to have information that is important to us
to keep tracking the graph (node execution result or chat history)

It can save the chat history like save the results of the LLMs every time or it can 
be complex and customize it

State is available for every node to access within our graph execution, we can persist it into
persistent storage. If I want to stop for some reason, the flow execution of the graph and
we want to continue from that same point with that given state, we can do it with persistence

We can build advanced concepts.

Cyclic Graph ~ in LangGraph we can implement loops (in langchain we cannot do it)

Human in the Loop ~ Get human feedback and this will decide where we need to go in the graph execution
(node A or node B)

Persistence ~ LangGraph has built-in function help us to persist our state of the graph.
make our software more robust and more fault-tolerant.


Now I will build ReAct Agent executer but with LangGraph
Deep dive into graph state and how to implement custom state

Implementing ReAct agent executor with LangGraph, and we can provide this agent executer with
tools (Tavily Search Tool)
"""

from dotenv import load_dotenv
import os


load_dotenv()



if __name__ == "__main__":
    print("Hello ReAct LangGraph with Function Calling")
    