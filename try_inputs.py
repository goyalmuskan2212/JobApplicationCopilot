from graph_v1 import graph

cases = {
    "1 synonym": {
        "job_text": "Must know JavaScript.",
        "my_skills": ["JS"],
    },
    "2 hidden words": {
        "job_text": "Digital marketing role. Contract laws are a plus.",
        "my_skills": ["Python"],
    },
    "3 spelling variant": {
        "job_text": "Experience with NodeJS and React required.",
        "my_skills": ["Node.js", "React"],
    },
}

for name, state in cases.items():
    result = graph.invoke(state)
    print(f"--- {name} ---")
    print("required:", result["required_skills"])
    print("matched: ", result["matched_skills"])
    print("fit:     ", result["fit_percent"], "%")
    print()