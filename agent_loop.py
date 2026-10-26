import os

from dotenv import load_dotenv

load_dotenv()  # must run before Langfuse and Groq read their keys

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langfuse import get_client
from langfuse.langchain import CallbackHandler

from retrieval import load_chunks, search

CHUNKS = load_chunks()


# ---------------------------------------------------------------
# 1) THE TOOL. The docstring is what the model reads to decide when
#    and how to use it, so write it like instructions.
# ---------------------------------------------------------------
@tool
def search_notes(query: str) -> str:
    """Search the candidate's project notes for evidence of a skill or experience.
    Use short keyword queries such as 'Docker' or 'REST API'.
    Returns matching note excerpts with citations, or says no evidence was found."""
    results = search(query, k=3, chunks=CHUNKS)
    if not results:
        return "No evidence found."
    return "\n\n".join(f"{chunk.citation}\n{chunk.text}" for _, chunk in results)


TOOLS = {"search_notes": search_notes}

SYSTEM_PROMPT = """You answer questions about a job candidate using ONLY the search_notes tool.
- Always search before answering.
- If a search finds nothing, try one or two different search terms before giving up.
- If there is still no evidence, say clearly that no evidence was found. Never invent experience.
- Cite evidence with the citation tags exactly as returned, for example [csv-importer.md#2].
- Keep the final answer under four sentences."""

# ---------------------------------------------------------------
# 2) THE MODEL, told which tools exist.
# ---------------------------------------------------------------
llm = ChatGroq(model=os.environ["GROQ_MODEL"], temperature=0)
llm_with_tools = llm.bind_tools([search_notes])

MAX_STEPS = 5


# ---------------------------------------------------------------
# 3) THE AGENT LOOP, written by hand.
# ---------------------------------------------------------------
def run_agent(question: str, config: RunnableConfig) -> str:
    messages = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=question)]

    for step in range(1, MAX_STEPS + 1):
        ai = llm_with_tools.invoke(messages, config=config)
        messages.append(ai)  # keep the model's reply in the history

        if not ai.tool_calls:  # no tool requested, so this is the final answer
            return ai.content

        for call in ai.tool_calls:
            print(f"  step {step}: model called {call['name']}({call['args']})")
            result = TOOLS[call["name"]].invoke(call["args"], config=config)
            # Send the result back, tagged with the call id.
            messages.append(ToolMessage(content=result, tool_call_id=call["id"]))

    return "Stopped: too many steps without a final answer."


if __name__ == "__main__":
    handler = CallbackHandler()
    agent = RunnableLambda(run_agent)  # wrapping it groups everything into ONE trace

    questions = [
        "Does the candidate have Docker experience? Give evidence.",
        "Does the candidate have Kubernetes experience?",
        "Does the candidate know about containers?",
    ]

    for q in questions:
        print(f"\nQUESTION: {q}")
        answer = agent.invoke(
            q, config={"callbacks": [handler], "run_name": "agent-loop-step3b", "tags": ["step3b"]}
        )
        print(f"ANSWER: {answer}")

    get_client().flush()  # short scripts exit before traces upload, so flush