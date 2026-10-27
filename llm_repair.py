import json

from langchain_core.exceptions import OutputParserException
from pydantic import ValidationError

MARKER = "from completion "


def find_schema(runnable):
    """Find the Pydantic class inside a with_structured_output chain."""
    last = getattr(runnable, "last", None)
    return getattr(last, "pydantic_object", None)


def completion_as_dict(error):
    """Recover the model's raw JSON answer from a parsing error."""
    raw = getattr(error, "llm_output", None)
    if not raw:
        text = str(error)
        start = text.find(MARKER)
        end = text.find(". Got:", start)
        if start != -1 and end != -1:
            raw = text[start + len(MARKER):end]
    try:
        return json.loads(raw) if raw else None
    except json.JSONDecodeError:
        return None


def unwrap_envelope(data):
    """Undo the wrapper {"description": "", "properties": {real fields}}."""
    while isinstance(data, dict) and isinstance(data.get("properties"), dict):
        data = data["properties"]
    return data


def repair(error: OutputParserException, schema):
    """Rescue a wrong-shape answer. Returns a validated object, or None."""
    if schema is None:
        return None
    data = completion_as_dict(error)
    if data is None:
        return None
    try:
        return schema.model_validate(unwrap_envelope(data))
    except ValidationError:
        return None  # still invalid after unwrapping: do not accept it