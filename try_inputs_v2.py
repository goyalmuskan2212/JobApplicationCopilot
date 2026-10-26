from graph_v2 import graph

cases = {
    "1 synonym": {
        "job_text": "Must know JavaScript.",
        "my_profile": "I build web apps with JS and React.",
    },
    "2 hidden words": {
        "job_text": "Digital marketing role. Contract laws are a plus.",
        "my_profile": "I write Python code.",
    },
    "3 spelling variant": {
        "job_text": "Experience with NodeJS and React required.",
        "my_profile": "I build apps with Node.js and React.",
    },
}

for name, state in cases.items():
    result = graph.invoke(state)
    print(f"--- {name} ---")
    print(result["report"])
    print()