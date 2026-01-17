"""Text preprocessing functions for README content."""

import re
from typing import Optional

import tiktoken


def clean_readme(text: str, remove_code: bool = True, remove_html: bool = True) -> str:
    """
    Clean README text by removing code blocks, HTML tags, and normalizing whitespace.

    Args:
        text: Raw README content
        remove_code: Whether to remove code blocks
        remove_html: Whether to remove HTML tags

    Returns:
        Cleaned text
    """
    if not text or not isinstance(text, str):
        return ""

    # Remove code blocks (```...```)
    if remove_code:
        text = re.sub(r"```[\s\S]*?```", "", text)
        text = re.sub(r"`[^`]+`", "", text)

    # Remove HTML tags
    if remove_html:
        text = re.sub(r"<[^>]+>", "", text)

    # Remove URLs
    text = re.sub(r"https?://\S+", "", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)
    text = text.strip()

    return text


def truncate_text(text: str, max_chars: int = 1000) -> str:
    """
    Truncate text to maximum number of characters.

    Args:
        text: Input text
        max_chars: Maximum characters to keep

    Returns:
        Truncated text
    """
    if not text:
        return ""
    return text[:max_chars]


def truncate_to_tokens(text: str, max_tokens: int = 3000, model: str = "gpt-4") -> str:
    """
    Truncate text to maximum number of tokens.

    Args:
        text: Input text
        max_tokens: Maximum tokens to keep
        model: Model name for tokenizer

    Returns:
        Truncated text
    """
    if not text:
        return ""

    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")

    tokens = encoding.encode(text)
    if len(tokens) <= max_tokens:
        return text

    truncated_tokens = tokens[:max_tokens]
    return encoding.decode(truncated_tokens)


def create_input_text(
    description: Optional[str],
    topics,
    readme: Optional[str],
    max_readme_chars: int = 1000
) -> str:
    """
    Create combined input text from repository metadata.

    Args:
        description: Repository description
        topics: Repository topics (string, list, or array)
        readme: README content
        max_readme_chars: Maximum characters for README

    Returns:
        Combined input text
    """
    import numpy as np

    parts = []

    if description and isinstance(description, str):
        parts.append(f"Description: {description}")

    # Handle topics - can be string, list, or numpy array
    if topics is not None:
        if isinstance(topics, str):
            if topics.strip():
                parts.append(f"Topics: {topics}")
        elif isinstance(topics, (list, np.ndarray)):
            # Convert array/list to comma-separated string
            topics_list = [str(t) for t in topics if t]
            if topics_list:
                parts.append(f"Topics: {', '.join(topics_list)}")

    if readme and isinstance(readme, str):
        cleaned_readme = clean_readme(readme)
        truncated_readme = truncate_text(cleaned_readme, max_readme_chars)
        if truncated_readme:
            parts.append(f"README: {truncated_readme}")

    return " | ".join(parts)
