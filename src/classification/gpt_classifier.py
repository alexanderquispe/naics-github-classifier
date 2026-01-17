"""GPT Classifier using GitHub Copilot API."""

import json
import os
import time
from typing import Dict, Optional, Tuple

import requests
from dotenv import load_dotenv

from .prompt_builder import build_classification_prompt

load_dotenv()


class GPTClassifier:
    """
    Classifier using GPT-4 via GitHub Copilot API.

    Attributes:
        api_endpoint: API endpoint URL
        model: Model name to use
        headers: Request headers with authentication
    """

    def __init__(
        self,
        api_endpoint: str = "https://api-model-lab.githubcopilot.com/chat/completions",
        model: str = "gpt-4.1",
        token: Optional[str] = None
    ):
        """
        Initialize GPT classifier.

        Args:
            api_endpoint: GitHub Copilot API endpoint
            model: Model name
            token: GitHub token (uses env var if not provided)
        """
        self.api_endpoint = api_endpoint
        self.model = model

        token = token or os.getenv("GITHUB_TOKEN")
        if not token:
            raise ValueError("GITHUB_TOKEN not found in environment")

        self.headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }

    def classify(
        self,
        readme_content: str,
        naics_code: str,
        naics_definition: str,
        max_retries: int = 5,
        retry_delay: float = 2.0,
        timeout: int = 60
    ) -> Tuple[bool, Dict]:
        """
        Classify a repository against a NAICS sector.

        Args:
            readme_content: Preprocessed README content
            naics_code: NAICS sector code
            naics_definition: Sector definition
            max_retries: Maximum retry attempts
            retry_delay: Delay between retries (seconds)
            timeout: Request timeout (seconds)

        Returns:
            Tuple of (success, result_dict)
            result_dict contains: match, score, rationale (or error)
        """
        prompt = build_classification_prompt(readme_content, naics_code, naics_definition)

        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1,
            "max_tokens": 200
        }

        for attempt in range(max_retries):
            try:
                response = requests.post(
                    self.api_endpoint,
                    headers=self.headers,
                    json=payload,
                    timeout=timeout
                )

                if response.status_code == 200:
                    result = response.json()
                    content = result["choices"][0]["message"]["content"]

                    # Parse JSON response
                    try:
                        parsed = json.loads(content)
                        return True, {
                            "match": parsed.get("match", False),
                            "score": parsed.get("score", 0),
                            "rationale": parsed.get("rationale", "")
                        }
                    except json.JSONDecodeError:
                        return False, {"error": f"Invalid JSON response: {content}"}

                elif response.status_code == 429:
                    # Rate limited - wait and retry
                    wait_time = retry_delay * (2 ** attempt)
                    time.sleep(wait_time)
                    continue

                else:
                    return False, {"error": f"API error {response.status_code}: {response.text}"}

            except requests.exceptions.Timeout:
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                return False, {"error": "Request timeout"}

            except Exception as e:
                return False, {"error": str(e)}

        return False, {"error": "Max retries exceeded"}

    def list_models(self) -> list:
        """
        List available models from the API.

        Returns:
            List of model names
        """
        models_endpoint = self.api_endpoint.replace("/chat/completions", "/models")

        try:
            response = requests.get(models_endpoint, headers=self.headers, timeout=30)
            if response.status_code == 200:
                return response.json().get("data", [])
        except Exception as e:
            print(f"Error listing models: {e}")

        return []
