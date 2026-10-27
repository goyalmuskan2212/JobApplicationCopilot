import os
from typing import Annotated, TypedDict

import groq
from dotenv import load_dotenv

load_dotenv()  # must run before Langfuse and Groq read their keys

from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_groq import ChatGroq
from langfuse import get_client
from langfuse.langchain import CallbackHandler
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from tools import search_notes

SYSTEM_PROMPT = """You answer questions about a job candidate using ONLY the search_notes tool.
- Always search before answering.
- If a search finds nothing, try one or two different search terms before giving up.
- If there is still no evidence, say clearly that no evidence was found. Never invent experience.
- Cite evidence with the citation tags exactly as returned, for example [csv-importer.md#2].
- Keep the final answer under four sentences."""

llm = ChatGroq(model=os.environ["GROQ_MODEL"], temperature=0)
llm_with_tools = llm.bind_tools([search_notes])

MAX_MODEL_ATTEMPTS = 3


def call_model(messages, config):
    """Call the model, retrying ONLY when it produces a malformed tool call."""
    for attempt in range(1, MAX_MODEL_ATTEMPTS + 1):
        try:
            return llm_with_tools.invoke(messages, config=config)
        except groq.BadRequestError as error:
            is_bad_tool_call = "tool_use_failed" in str(error)
            if not is_bad_tool_call or attempt == MAX_MODEL_ATTEMPTS:
                raise  # a different error, or out of attempts: let it surface
            print(f"  (malformed tool call from the model; retrying, attempt {attempt + 1} of {MAX_MODEL_ATTEMPTS})")


# 1) STATE: the conversation so far. add_messages is a REDUCER: when a node
#    returns {"messages": [msg]}, LangGraph APPENDS it instead of replacing the list.
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


# 2) NODES
def agent(state: State, config: RunnableConfig) -> dict:
    # The system prompt is prepended on every call, not stored in the state.
    reply = call_model([SystemMessage(content=SYSTEM_PROMPT)] + state["messages"], config)
    return {"messages": [reply]}


tools_node = ToolNode([search_notes])  # runs every tool the last AI message asked for


# 3) CONDITIONAL EDGE: a function that returns the NAME of the next node.
def route_after_agent(state: State) -> str:
    last_message = state["messages"][-1]
    return "tools" if last_message.tool_calls else END


# 4) GRAPH
builder = StateGraph(State)
builder.add_node("agent", agent)
builder.add_node("tools", tools_node)

builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", route_after_agent, {"tools": "tools", END: END})
builder.add_edge("tools", "agent")  # this edge makes the cycle

graph = builder.compile()


if __name__ == "__main__":
    handler = CallbackHandler()

    questions = [
        "Does the candidate have Docker experience? Give evidence.",
        "Does the candidate have Kubernetes experience?",
        "Does the candidate know about containers?",
    ]

    for q in questions:
        print(f"\nQUESTION: {q}")
        try:
            result = graph.invoke(
                {"messages": [HumanMessage(content=q)]},
                config={
                    "callbacks": [handler],
                    "run_name": "agent-graph-step3c",
                    "tags": ["step3c"],
                    "recursion_limit": 8,
                },
            )
        except GraphRecursionError:
            print("Stopped: hit the recursion limit.")
            continue

        # Show the whole message trail: this list IS the agent's memory.
        for m in result["messages"]:
            if m.type == "ai" and m.tool_calls:
                print(f"  [ai] asked for: {[(c['name'], c['args']) for c in m.tool_calls]}")
            else:
                print(f"  [{m.type}] {str(m.content)[:100]!r}")
        print("ANSWER:", result["messages"][-1].content)

    print("\nGraph shape (paste into mermaid.live):")
    print(graph.get_graph().draw_mermaid())

    get_client().flush()  # short scripts exit before traces upload, so flush