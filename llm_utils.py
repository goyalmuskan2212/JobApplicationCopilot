import groq
from langchain_core.exceptions import OutputParserException

from llm_repair import find_schema, repair

RETRYABLE_API_CODES = ("tool_use_failed", "json_validate_failed")


def invoke_with_retry(runnable, messages, config, attempts=3):
    """Call a model; repair a wrong-shape answer, retry only malformed output."""
    schema = find_schema(runnable)
    for attempt in range(1, attempts + 1):
        try:
            return runnable.invoke(messages, config=config)
        except OutputParserException as error:
            fixed = repair(error, schema)
            if fixed is not None:
                print("  (unwrapped the schema envelope and validated the answer)")
                return fixed
            if attempt == attempts:
                raise
            print(f"  (wrong shape; retrying, attempt {attempt + 1} of {attempts})")
        except groq.BadRequestError as error:
            retryable = any(c in str(error) for c in RETRYABLE_API_CODES)
            if not retryable or attempt == attempts:
                raise
            print(f"  (output rejected; retrying, attempt {attempt + 1} of {attempts})")