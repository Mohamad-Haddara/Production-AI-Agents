"""
Hands-on: Map-Reduce Strategy

Parallel Agent Execution in LangGraph
Running multiple agents simultaneously, not one after another.

Instead of running sequential in our multi-agent flow, we can have also parallel agent execution.

"""

from re import S
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
import asyncio
from dotenv import load_dotenv


load_dotenv()


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

# 1) Create state
class ParallelState(TypedDict):
    query: str
    research_result: str
    creative_result: str
    technical_result: str
    final_synthesize: str



def create_parrallel_research():
    """Three research agents working in parrallel."""

    def research_agent(state: ParallelState) -> dict:
        """Academic/factual research"""

        response = llm.invoke([
            SystemMessage(content="You are an academic researcher. Provide factual information."),
            HumanMessage(content=f"Research this topic: {state['query']}")
        ])

        return {"research_result": response.content}

    
    def creative_agent(state: ParallelState) -> dict:
        """Creative perspective"""

        response = llm.invoke([
            SystemMessage(content="You are creative thinker. Provide novel perspectives"),
            HumanMessage(content=f"Give creative insight on: {state['query']}")
        ])

        return {"creative_result": response.content}

    

    def technical_agent(state: TypedDict) -> dict:
        """technical analysis"""

        response = llm.invoke([
            SystemMessage(content="You are technical analyst. Provide practical, implementation"),
            HumanMessage(content=f"Analyze technically: {state['query']}")
        ])

        return {
            "technical_result": response.content
        }


    # Synthesizer ~ one that synthesizer
    def synthesize(state: ParallelState) -> dict:
        """Combine all perspectives."""
        synthesize_prompt = f"""
            Synthesize these three perspective into comprehensive result.

            RESEARCH: {state['research_result']}

            CREATIVE: {state['creative_result']}

            TECHNICAL: {state['technical_result']}

            Create a unified, well-structured response. 

        """

        response = llm.invoke([
            SystemMessage(content="You are an expert synthesizer. Combine multiple perspective into coherent insights"),
            HumanMessage(content=synthesize_prompt)
        ])

        return {
            "final_synthesize": response.content
        }

    
    graph = StateGraph(ParallelState)


    graph.add_node("research", research_agent)
    graph.add_node("creative", creative_agent)
    graph.add_node("technical", technical_agent)
    graph.add_node("synthesize", synthesize)

    # Parallel Execution
    graph.add_edge(START, "research")
    graph.add_edge(START, "creative")
    graph.add_edge(START, "technical")

    graph.add_edge("research", "synthesize")
    graph.add_edge("technical", "synthesize")
    graph.add_edge("creative", "synthesize")
    
    graph.add_edge("synthesize", END)

    return graph.compile()



def parallel_exection():
    """Parallel agent execution."""

    agent = create_parrallel_research()

    print("Parallel Agent Execution:\n")

    result = agent.invoke({
        "query": "The future remote work",
        "research_result": "",
        "creative_result": "",
        "technical_result": "",
        "final_synthesize": ""
    })

    print("Individual Perspectives:")
    print(f"\n[Research]\n{result['research_result'][:300]}...")
    print(f"\n[Creative]\n{result['creative_result'][:300]}...")
    print(f"\n[Technical]\n{result['technical_result'][:300]}...")

    print(f"\n{"="*50}")
    print(f"[SYTHESIZED]\n{result['final_synthesize']}\n")

# Map-Reduce Strategy ~ Full document summarization. 
# Architecture: Process each document, then map each document, then we are going to pass through map-reducer - which will combine all summaries and then we hit the end of the workflow

# Map-Reduce Pattern
class MapReduceState(TypedDict):
    documents: list[str] # Input N documents ~ number of input are dynamic
    summaries: list[str] 
    final_summary: str


def map_reduce_summarizer():
    """Summarize multiple documents in parallel."""

    def map_summarize(state: MapReduceState) -> dict:
        """Summarize each document (runs in parallel for each)"""
        summaries = []

        for doc in state['documents']:
            response = llm.invoke([
                SystemMessage(content= "Summarize this document in 2-3 sentences"),
                HumanMessage(content = doc)
            ])

            summaries.append(response.content)

        return {"summaries": summaries}
    
    def reduce_combine(state: MapReduceState) -> dict:
        """Combine all summaries"""
        # many summary become one summary ~ take all N summaries
        all_summaries = "\n\n".join([
           f"Summary {i+1}: {s}" for i, s in enumerate(state['summaries'])
        ])

        # Take all summaries and ask llm to produce on coherent overview
        response = llm.invoke([
            SystemMessage(content= "Combine all these summaries into one coherent overview"),
            HumanMessage(content=all_summaries)
        ])

        return {"final_summary": response.content}
        """
        Map-Reduce means ~ N input and 1 output
        """

    graph = StateGraph(MapReduceState)

    graph.add_node("map", map_summarize)
    graph.add_node("reduce", reduce_combine)

    graph.add_edge(START, "map")
    graph.add_edge("map", "reduce")
    graph.add_edge("reduce", END)


    return graph.compile()


def map_reduce():
    """map-reduce pattern"""

    agent = map_reduce_summarizer()

    documents = [
        "Python is a high-level programming language known for its simplicity and readability",
        "Machine learning is a subset of AI that enables systems to learn from data.",
        "Cloud computing provide on-demand access to computing service. Major providers are Azure, AWS, and GCP."
    ]

    print("\nMap-Reduce Summarization:\n")

    result = agent.invoke({
        "documents":documents,
        "summaries":[],
        "final_summary":""
    })

    print("Individual summaries")
    for i, summary in enumerate(result['summaries']):
        print(f"\n {i+1} - Summaries: {result['summaries']}\n")
    
    print(f"\n[final summary]: {result['final_summary']}\n")

"""
We might summaries 500 documents -> This will exceed context window.

Production fix will call hierarchical reduce which is going to summaries in batches in 10 - this is very important
It will summaries those summaries, then it will summaries those again.

So Map-reduce -> map - reduce - reduce - reduce...

Thats how production at scale works
"""
if __name__ == "__main__":
    #parallel_exection()
    map_reduce()