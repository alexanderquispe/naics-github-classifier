"""Prompt construction for GPT classification."""

import json
from pathlib import Path
from typing import Optional


def build_classification_prompt(
    readme_content: str,
    naics_code: str,
    naics_definition: str
) -> str:
    """
    Generate classification prompt for GitHub repository NAICS sector assignment.

    This function creates a structured prompt for accurately classifying GitHub repositories
    into one of 20 NAICS industry sectors based on README content analysis.

    Args:
        readme_content: Preprocessed README content
        naics_code: Target NAICS sector code for classification
        naics_definition: Description of the NAICS sector (semicolon-separated industries)

    Returns:
        Formatted prompt string for LLM classification
    """
    prompt = f"""
TASK: GitHub Repository NAICS Sector Classification

You are a domain expert tasked with classifying GitHub repositories into NAICS industry sectors for GitHub's repository categorization system. Your goal is to determine with high precision whether this repository belongs to NAICS Sector {naics_code}.

REPOSITORY README CONTENT:
<readme>
{readme_content.strip()}
</readme>

TARGET NAICS SECTOR:
<naics_sector>
Sector {naics_code}: {naics_definition.strip()}
</naics_sector>

CLASSIFICATION FRAMEWORK:

A repository should be classified as "Yes" for this sector if it demonstrates CLEAR ALIGNMENT with one or more of these criteria:

1. **Industry-Specific Software**: Applications, tools, or systems designed specifically for use within this industry sector
   - Example: Farm management software for Agriculture (Sector 11)
   - Example: Church management systems for Religious Organizations (Sector 81)

2. **Sector-Relevant Functionality**: Code that implements processes, calculations, or workflows specific to this industry
   - Example: Prayer time calculators for Religious Organizations
   - Example: Crop yield prediction models for Agriculture

3. **Industry Domain Applications**: Software that directly serves businesses, organizations, or activities within this sector
   - Example: Restaurant POS systems for Food Services (Sector 72)
   - Example: Educational platforms for Educational Services (Sector 61)

4. **Sector-Specific Data/Research**: Datasets, analysis tools, or research implementations focused on this industry
   - Example: Agricultural sensor data analysis
   - Example: Healthcare outcome prediction models

CLASSIFICATION STANDARDS:

**INCLUDE ("Yes") when:**
- The repository's primary purpose aligns with the sector
- The software would be used by businesses/organizations in this sector
- The code implements sector-specific functionality or processes
- The project addresses sector-specific problems or use cases

**EXCLUDE ("No") when:**
- The repository serves multiple sectors equally (generic tools)
- Industry connection is only tangential or in examples
- The primary use case is outside this sector
- No clear business or operational relevance to the sector

SCORING GUIDE:
- 9-10: Core industry software with direct sector application
- 7-8: Strong sector relevance with clear industry use cases
- 5-6: Moderate sector connection with identifiable applications
- 3-4: Weak sector relevance, mostly tangential
- 1-2: No meaningful sector connection

ANALYSIS REQUIREMENTS:
1. Identify the repository's primary purpose and functionality
2. Assess alignment with the target NAICS sector
3. Determine the most applicable classification criterion
4. Consider practical usage within the sector

Provide your response in this exact JSON format:
{{
    "match": true or false,
    "score": 1-10,
    "rationale": "Concise explanation of classification decision, including primary repository purpose, specific sector alignment criteria met, and justification for inclusion/exclusion"
}}

IMPORTANT: Base your decision on the repository's PRIMARY purpose and DIRECT applicability to the sector. Be precise and consistent in your classifications.
"""
    return prompt


def load_naics_definitions(json_path: str) -> dict:
    """
    Load NAICS sector definitions from JSON file.

    Args:
        json_path: Path to the NAICS definitions JSON file

    Returns:
        Dictionary mapping NAICS codes to their definitions
    """
    path = Path(json_path)
    if not path.exists():
        raise FileNotFoundError(f"NAICS JSON file not found at {json_path}")

    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def build_prompt_with_lookup(
    readme_content: str,
    naics_code: str,
    naics_json_path: str
) -> str:
    """
    Build classification prompt by looking up NAICS definition from JSON file.

    Args:
        readme_content: Preprocessed README content
        naics_code: Target NAICS sector code
        naics_json_path: Path to NAICS definitions JSON

    Returns:
        Formatted prompt string

    Raises:
        ValueError: If NAICS code not found in definitions
    """
    naics_data = load_naics_definitions(naics_json_path)

    if naics_code not in naics_data:
        raise ValueError(f"Sector code {naics_code} not found in NAICS definitions.")

    naics_definition = naics_data[naics_code]

    if not naics_definition or naics_definition.strip() == "":
        raise ValueError(f"No industry descriptions found for sector {naics_code}.")

    return build_classification_prompt(readme_content, naics_code, naics_definition)
