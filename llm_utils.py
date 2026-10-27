import groq

# Groq error codes that mean "the model produced malformed output".
RETRYABLE = ("tool_use_failed", "json_validate_failed")


def invoke_with_retry(runnable, messages, config, attempts: int = 3):
    """Call a model, retrying ONLY when it produced malformed output."""
    for attempt in range(1, attempts + 1):
        try:
            return runnable.invoke(messages, config=config)
        except groq.BadRequestError as error:
            retryable = any(code in str(error) for code in RETRYABLE)
            if not retryable or attempt == attempts:
                raise  # a different error, or out of attempts: let it surface
            print(f"  (model output rejected; retrying, attempt {attempt + 1} of {attempts})")