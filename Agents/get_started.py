from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from API_integration.GENERAL_KEYS import TAVILY_API_KEY
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
import os

os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY

# Create agent
memory = MemorySaver()
#model = ChatAnthropic(model_name="claude-3-sonnet-20240229")
model = ChatOpenAI(model="gpt-4o-mini")
search = TavilySearchResults(max_results=2)
tools = [search]
agent_executor = create_react_agent(model, tools, checkpointer=memory)

# use the agent
# agent session
config = {"configurable": {"thread_id": "abc123"}}

for chunk in agent_executor.stream(
        {"messages": [HumanMessage(content="hi im alex, i live in romania")]}, config
):
    print(chunk)
    print("----")

for chunk in agent_executor.stream(
        {"messages": [HumanMessage(content="whats the weather where i live?")]}, config
):
    print(chunk)
    print("----")
