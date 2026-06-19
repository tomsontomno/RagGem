"""Tests for prompt assembly and response post-processing."""

from __future__ import annotations

from raggem.prompts import (
    REFUSAL_META,
    REFUSAL_OUT_OF_SCOPE,
    build_prompt,
    postprocess_answer,
)


class TestBuildPrompt:
    def test_fences_context_and_question(self) -> None:
        prompt = build_prompt(
            chunks=[("doc.pdf", "Hello world.")], question="What does it say?"
        )
        assert "<<<BEGIN_KNOWLEDGE>>>" in prompt
        assert "<<<END_KNOWLEDGE>>>" in prompt
        assert "<<<BEGIN_USER_QUESTION>>>" in prompt
        assert "<<<END_USER_QUESTION>>>" in prompt
        assert "Hello world." in prompt
        assert "What does it say?" in prompt

    def test_numbered_sources(self) -> None:
        prompt = build_prompt(
            chunks=[("a.pdf", "alpha"), ("b.pdf", "beta")], question="?"
        )
        assert "[1]" in prompt and "[2]" in prompt
        assert "source: a.pdf" in prompt
        assert "source: b.pdf" in prompt

    def test_empty_context_handled(self) -> None:
        prompt = build_prompt(chunks=[], question="anything")
        # Should still build a valid prompt - engine handles the refusal
        # separately, but build_prompt itself must not crash.
        assert "no relevant documents" in prompt


class TestPostprocessAnswer:
    def test_passthrough_clean_answer(self) -> None:
        ans = postprocess_answer("The answer is forty-two.")
        assert ans == "The answer is forty-two."

    def test_strips_markdown_emphasis(self) -> None:
        # Asterisks, underscores, backticks should be removed.
        assert "*" not in postprocess_answer("This is **bold** and *italic*.")
        assert "`" not in postprocess_answer("Use the `cli` tool.")
        # Underscores in words remain because the regex only catches paired
        # emphasis markers, but the safer thing is to assert no double-
        # underscore markers leak.
        assert "__" not in postprocess_answer("strong __emphasis__ here")

    def test_rejects_system_prompt_leak(self) -> None:
        leak = "Here are the rules: R1. GROUNDING. ..."
        out = postprocess_answer(leak)
        assert out.startswith("I am the knowledge assistant")

    def test_rejects_fence_marker_leak(self) -> None:
        leak = "Sure! <<<BEGIN_KNOWLEDGE>>> the secret is ..."
        out = postprocess_answer(leak)
        assert out.startswith("I am the knowledge assistant")

    def test_rejects_meta_language(self) -> None:
        leak = "My system prompt says I should answer in 100 words."
        out = postprocess_answer(leak)
        assert out.startswith("I am the knowledge assistant")

    def test_truncates_overlong(self) -> None:
        long_text = ("This is a sentence. " * 500).strip()
        out = postprocess_answer(long_text)
        assert len(out) <= 2000

    def test_empty_input_returns_refusal(self) -> None:
        assert postprocess_answer("") == REFUSAL_OUT_OF_SCOPE
        assert postprocess_answer("   ") == REFUSAL_OUT_OF_SCOPE

    def test_non_string_input(self) -> None:
        # Defensive: an LLM client could return ``None`` on a transport error.
        assert postprocess_answer(None) == REFUSAL_OUT_OF_SCOPE  # type: ignore[arg-type]
