from typing import TypedDict

from dotenv import load_dotenv

load_dotenv()  # must run before Langfuse reads its keys

from langfuse import get_client
from langfuse.langchain import CallbackHandler
from langgraph.graph import END, START, StateGraph

# A tiny vocabulary of skills to look for. Step 2 replaces this with an LLM.
KNOWN_SKILLS = [
    "python", "typescript", "javascript", "react", "node.js", "sql",
    "docker", "aws", "machine learning", "langgraph", "fastapi", "git",
]


# 1) STATE: the shared notebook for one run.
class State(TypedDict):
    job_text: str               # input: the job posting
    my_skills: list[str]        # input: your skills
    required_skills: list[str]  # filled by extract_required_skills
    matched_skills: list[str]   # filled by compare_with_profile
    missing_skills: list[str]   # filled by compare_with_profile
    fit_percent: int            # filled by compare_with_profile
    report: str                 # filled by build_report


# 2) NODES: plain functions. Take the state, return ONLY what changed.
def extract_required_skills(state: State) -> dict:
    text = state["job_text"].lower()
    found = [skill for skill in KNOWN_SKILLS if skill in text]
    return {"required_skills": found}


def compare_with_profile(state: State) -> dict:
    required = state["required_skills"]
    mine = [s.lower() for s in state["my_skills"]]
    matched = [s for s in required if s in mine]
    missing = [s for s in required if s not in mine]
    percent = round(100 * len(matched) / len(required)) if required else 0
    return {
        "matched_skills": matched,
        "missing_skills": missing,
        "fit_percent": percent,
    }


def build_report(state: State) -> dict:
    report = (
        f"Fit: {state['fit_percent']}%\n"
        f"Matched: {', '.join(state['matched_skills']) or 'none'}\n"
        f"Missing: {', '.join(state['missing_skills']) or 'none'}"
    )
    return {"report": report}


# 3) GRAPH: register the nodes, then connect them with edges.
builder = StateGraph(State)

builder.add_node("extract_required_skills", extract_required_skills)
builder.add_node("compare_with_profile", compare_with_profile)
builder.add_node("build_report", build_report)

builder.add_edge(START, "extract_required_skills")
builder.add_edge("extract_required_skills", "compare_with_profile")
builder.add_edge("compare_with_profile", "build_report")
builder.add_edge("build_report", END)

# 4) COMPILE: validates the graph and makes it runnable.
graph = builder.compile()


if __name__ == "__main__":
    handler = CallbackHandler()  # Langfuse's listener

    job_posting = """
    We are hiring a backend engineer. You should be strong in Python and SQL,
    have used Docker and AWS, and know FastAPI. Machine learning is a plus.
    """

    # 5) INVOKE: run the graph once with the starting state.
    result = graph.invoke(
        {
            "job_text": job_posting,
            "my_skills": ["Python", "TypeScript", "React", "Node.js", "SQL", "Git", "Machine Learning"],
        },
        config={"callbacks": [handler], "run_name": "job-copilot-step1", "tags": ["step1"]},
    )

    print(result["report"])
    print("\nFull final state:")
    print(result)

    print("\nGraph shape (paste into mermaid.live to see a picture):")
    print(graph.get_graph().draw_mermaid())

    get_client().flush()  # short scripts exit before traces upload, so flush