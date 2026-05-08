"""STT HTTP response parsing (OpenAI-style JSON)."""

from book2audio.stt_parakeet import coerce_stt_response_to_text


def test_coerce_openai_style():
    assert coerce_stt_response_to_text({"text": "  hello  "}) == "hello"


def test_coerce_segments():
    payload = {"segments": [{"text": "a"}, {"text": "b"}]}
    assert coerce_stt_response_to_text(payload) == "a b"


def test_coerce_empty():
    assert coerce_stt_response_to_text({}) is None
    assert coerce_stt_response_to_text(None) is None
