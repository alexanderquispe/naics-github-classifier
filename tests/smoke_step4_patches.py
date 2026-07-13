"""
Smoke test for the step-4 patches.

Verifies:
  1. The patched prompt_builder + gpt_classifier modules import without error.
  2. The rendered prompt contains the nested "NAICS {code}" JSON schema and
     the string-typed score/match placeholders.
  3. The gpt_classifier payload contains a system message, temperature=0,
     and max_tokens=3000 (without hitting the API).
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import classification.prompt_builder as pb  # noqa: E402
import classification.gpt_classifier as gc  # noqa: E402


def check_prompt():
    prompt = pb.build_classification_prompt(
        readme_content="description: ML lib, topics: pytorch, readme: ...",
        naics_code="51",
        naics_definition="Software Publishers; Computing Infrastructure Providers",
    )
    assert '"NAICS 51"' in prompt, "Missing nested NAICS {code} key"
    assert '"score": "1-10"' in prompt, "Score should be string placeholder"
    assert '"match": "Yes" or "No"' in prompt, "Match should be string Yes/No"
    print("[PASS] Prompt contains nested NAICS-keyed JSON schema with string types.")


def check_payload():
    # Stub out the env so the classifier initialises without a real key
    import os
    os.environ["OPENAI_API_KEY"] = "sk-fake-key-for-smoke-test"

    classifier = gc.GPTClassifier(backend="openai")

    # Reach inside classify() to inspect the payload without making a request.
    # We monkeypatch requests.post to capture the JSON payload and return a
    # canned 200 OK so the function returns normally.
    captured = {}

    class FakeResponse:
        status_code = 200
        text = ""
        def json(self):
            return {
                "choices": [
                    {
                        "message": {
                            "content": (
                                '{"NAICS 51": {"rationale": "stub", '
                                '"score": "8", "match": "Yes"}}'
                            )
                        }
                    }
                ]
            }

    def fake_post(url, headers=None, json=None, timeout=None):
        captured.update(json)
        return FakeResponse()

    import classification.gpt_classifier as gc_mod
    original = gc_mod.requests.post
    gc_mod.requests.post = fake_post
    try:
        success, result = classifier.classify(
            readme_content="ML library for transformers",
            naics_code="51",
            naics_definition="Software Publishers",
        )
    finally:
        gc_mod.requests.post = original

    assert success is True, f"classify() should succeed on a 200 with valid JSON; got {result}"
    assert captured["temperature"] == 0, f"temperature should be 0; got {captured.get('temperature')}"
    assert captured["max_tokens"] == 3000, f"max_tokens should be 3000; got {captured.get('max_tokens')}"
    assert captured["messages"][0]["role"] == "system", "First message must be system"
    sys_content = captured["messages"][0]["content"]
    assert "NAICS industry 51" in sys_content, f"System msg should mention NAICS 51; got: {sys_content}"
    assert result["match"] == "Yes", f"Parser should return string 'Yes'; got {result['match']!r}"
    assert result["score"] == "8", f"Parser should return string '8'; got {result['score']!r}"
    assert result["rationale"] == "stub", f"Parser should return the rationale; got {result['rationale']!r}"
    print("[PASS] Payload has system message, temperature=0, max_tokens=3000.")
    print("[PASS] Parser correctly unwraps nested 'NAICS {code}' key with string types.")


def main():
    check_prompt()
    check_payload()
    print("All step-4 patches verified.")


if __name__ == "__main__":
    main()
