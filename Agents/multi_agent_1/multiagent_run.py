from langgraph.graph import END, StateGraph
from orchestration import AgentState, research_node, chart_node, tool_node, router
from langchain_community.callbacks.manager import get_openai_callback
from langchain_core.messages import HumanMessage


workflow= StateGraph(AgentState)

workflow.add_node("Researcher", research_node)
workflow.add_node("Chart Generator", chart_node)
workflow.add_node("call_tool", tool_node)

workflow.add_conditional_edges(
    "Researcher",
    router,
    {"continue": "Chart Generator", "call_tool": "call_tool", "end": END},
)
workflow.add_conditional_edges(
    "Chart Generator",
    router,
    {"continue": "Researcher", "call_tool":"call_tool", "end": END},
)
workflow.add_conditional_edges(
    "call_tool",
    # agent node updates 'sender', tool calling dont
    lambda  x:x["sender"],
    {
        "Researcher": "Researcher",
        "Chart Generator": "Chart Generator",
    },
)
workflow.set_entry_point("Researcher")
graph=workflow.compile()


with get_openai_callback() as cb:
    for s in graph.stream(
            {
                "messages": [
                    HumanMessage(
                        content="Fetch the Romania's GDP over the past 5 years,"
                                "then draw a line graph of it."
                                "Once you code it up, finish."
                    )
                ],
            },
            {"recursion_limit": 150},
    ):
        print(s)
        print("----")

    print(f"Total Tokens: {cb.total_tokens}")
    print(f"Prompt Tokens: {cb.prompt_tokens}")
    print(f"Completion Tokens: {cb.completion_tokens}")
    print(f"Total Cost (USD): ${cb.total_cost}")