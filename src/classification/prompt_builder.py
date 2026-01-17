"""Prompt construction for GPT classification."""

from typing import Dict


def build_classification_prompt(
    readme_content: str,
    naics_code: str,
    naics_definition: str
) -> str:
    """
    Build a classification prompt for GPT.

    Args:
        readme_content: Preprocessed README content
        naics_code: NAICS sector code
        naics_definition: Description of the NAICS sector

    Returns:
        Formatted prompt string
    """
    prompt = f"""You are an expert industry classifier. Your task is to determine whether a GitHub repository belongs to a specific NAICS (North American Industry Classification System) sector.

## Repository README:
{readme_content}

## Target NAICS Sector:
- **Code**: {naics_code}
- **Definition**: {naics_definition}

## Classification Framework:
Evaluate the repository based on these criteria:
1. **Industry-Specific Software**: Does this repo develop tools/software specifically for this industry?
2. **Sector-Relevant Functionality**: Does the core functionality serve this sector's needs?
3. **Domain Applications**: Is this repository used for applications within this industry?
4. **Sector-Specific Data**: Does it process, analyze, or generate data relevant to this sector?

## Scoring Guidelines:
- **9-10**: Strong, direct industry match - clearly serves this specific sector
- **7-8**: Good match - significant relevance to the sector
- **5-6**: Moderate match - some relevance but not sector-specific
- **3-4**: Weak match - tangential connection to the sector
- **1-2**: No match - unrelated to this sector

## Response Format:
Respond ONLY with a valid JSON object:
{{
    "match": true/false,
    "score": <1-10>,
    "rationale": "<brief explanation in 1-2 sentences>"
}}
"""
    return prompt


def build_batch_prompt(repos_batch: list, naics_code: str, naics_definition: str) -> str:
    """
    Build a prompt for batch classification (multiple repos at once).

    Args:
        repos_batch: List of (repo_id, readme_content) tuples
        naics_code: NAICS sector code
        naics_definition: Description of the NAICS sector

    Returns:
        Formatted batch prompt
    """
    # For batch processing, we still do one at a time for accuracy
    # This function is a placeholder for potential batch optimization
    raise NotImplementedError("Batch prompting not yet implemented")
