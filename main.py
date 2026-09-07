from dotenv import load_dotenv
from importlib.metadata import version

load_dotenv()


core_version = version("langchain-core")
lg_version = version("langgraph")
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic


print(f"Langchain-core version: {core_version}") # 1.6.2
print(f"Langgraph-core version: {lg_version}") # 1.2.11



def main():
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0
    )

    response = llm.invoke("Say setup complete in one word")
    print(f"Response from ChatOpenAI: {response}")

    print("Setup complete")

if __name__ == "__main__":
    main()
