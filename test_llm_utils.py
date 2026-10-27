from langchain_core.exceptions import OutputParserException

from graph_v2 import JobRequirements
from llm_utils import repair


def test_unwraps_schema_envelope():
    bad = (
        '{"description": "", "properties": {"job_title": "Platform Engineer", '
        '"requirements": [{"skill": "Kubernetes", "importance": "must_have"}]}}'
    )
    error = OutputParserException("Failed to parse", llm_output=bad)
    fixed = repair(error, JobRequirements)
    assert fixed is not None
    assert fixed.job_title == "Platform Engineer"
    assert fixed.requirements[0].skill == "Kubernetes"


def test_gives_up_on_garbage():
    error = OutputParserException("Failed to parse", llm_output="not json at all")
    assert repair(error, JobRequirements) is None


def test_rejects_wrong_content_even_after_unwrapping():
    bad = '{"description": "", "properties": {"unrelated": 1}}'
    error = OutputParserException("Failed to parse", llm_output=bad)
    assert repair(error, JobRequirements) is None


if __name__ == "__main__":
    test_unwraps_schema_envelope()
    test_gives_up_on_garbage()
    test_rejects_wrong_content_even_after_unwrapping()
    print("all tests passed")