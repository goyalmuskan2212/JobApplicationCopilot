import groq
from langchain_core.exceptions import OutputParserException

# Groq error codes that mean "the model produced malformed output".
RETRYABLE_API_CODES = ("tool_use_failed", "json_validate_failed")


def invoke_with_retry(runnable, messages, config, attempts: int = 3):
    """Call a model, retrying ONLY when its output was malformed or the wrong shape."""
    for attempt in range(1, attempts + 1):
        try:
            return runnable.invoke(messages, config=config)
        except OutputParserException:
            # The model answered, but not in the schema's shape.
            if attempt == attempts:
                raise
            print(f"  (model answered in the wrong shape; retrying, attempt {attempt + 1} of {attempts})")
        except groq.BadRequestError as error:
            retryable = any(code in str(error) for code in RETRYABLE_API_CODES)
            if not retryable or attempt == attempts:
                raise  # a different error, or out of attempts: let it surface
            print(f"  (model output rejected; retrying, attempt {attempt + 1} of {attempts})")