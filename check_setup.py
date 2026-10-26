import os
import sys

from dotenv import load_dotenv

load_dotenv()  # reads .env into environment variables


def check(label: str, ok: bool) -> None:
    print(f"[{'OK' if ok else 'FAIL'}] {label}")
    if not ok:
        sys.exit(1)


# 1) Python version
check("Python 3.11+", sys.version_info >= (3, 11))

# 2) All keys present
required = [
    "GROQ_API_KEY", "GROQ_MODEL",
    "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_HOST",
]
for key in required:
    check(f"{key} is set in .env", bool(os.getenv(key)))

# 3) Langfuse credentials are valid
from langfuse import get_client

langfuse = get_client()
check("Langfuse authentication", langfuse.auth_check())

# 4) Groq answers, and the call is traced to Langfuse
from langchain_groq import ChatGroq
from langfuse.langchain import CallbackHandler

llm = ChatGroq(model=os.environ["GROQ_MODEL"], temperature=0)
handler = CallbackHandler()

reply = llm.invoke(
    "In one short sentence, what is a job description?",
    config={"callbacks": [handler]},
)
print("Groq says:", reply.content)
check("Groq call returned text", bool(reply.content))

langfuse.flush()  # short scripts exit before traces upload, so flush
print("\nAll good. Open Langfuse, then Tracing, and look for your trace.")