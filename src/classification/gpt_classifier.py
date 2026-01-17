"""GPT Classifier supporting both OpenAI API and GitHub Copilot API."""

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
    Classifier using GPT-4 via OpenAI API or GitHub Copilot API.

    Supports two backends:
    - "openai": Standard OpenAI API (requires OPENAI_API_KEY)
    - "github": GitHub Copilot API (requires GITHUB_TOKEN)

    Attributes:
        api_endpoint: API endpoint URL
        model: Model name to use
        headers: Request headers with authentication
        backend: Which API backend to use ("openai" or "github")
    """

    # API configurations
    BACKENDS = {
        "openai": {
            "endpoint": "https://api.openai.com/v1/chat/completions",
            "env_var": "OPENAI_API_KEY",
            "default_model": "gpt-4-turbo",
            "models": ["gpt-4-turbo", "gpt-4", "gpt-3.5-turbo", "gpt-4o", "gpt-4o-mini"]
        },
        "github": {
            "endpoint": "https://api-model-lab.githubcopilot.com/chat/completions",
            "env_var": "GITHUB_TOKEN",
            "default_model": "gpt-4.1",
            "models": ["gpt-4.1", "gpt-4", "gpt-3.5-turbo"]
        }
    }

    def __init__(
        self,
        backend: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None
    ):
        """
        Initialize GPT classifier.

        Args:
            backend: API backend to use ("openai" or "github").
                     Auto-detects based on available env vars if not specified.
            model: Model name. Uses backend default if not specified.
            api_key: API key. Uses env var if not specified.
        """
        # Auto-detect backend if not specified
        if backend is None:
            backend = self._detect_backend()

        if backend not in self.BACKENDS:
            raise ValueError(f"Invalid backend: {backend}. Choose from: {list(self.BACKENDS.keys())}")

        self.backend = backend
        config = self.BACKENDS[backend]

        # Set endpoint
        self.api_endpoint = config["endpoint"]

        # Set model
        self.model = model or config["default_model"]

        # Get API key
        token = api_key or os.getenv(config["env_var"])
        if not token:
            raise ValueError(
                f"{config['env_var']} not found in environment.\n"
                f"Set it in .env file or pass api_key parameter.\n"
                f"Backend: {backend}"
            )

        self.headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }

        print(f"GPTClassifier initialized:")
        print(f"  Backend: {backend}")
        print(f"  Model: {self.model}")
        print(f"  Endpoint: {self.api_endpoint}")

    def _detect_backend(self) -> str:
        """Auto-detect which backend to use based on available env vars."""
        if os.getenv("OPENAI_API_KEY"):
            return "openai"
        elif os.getenv("GITHUB_TOKEN"):
            return "github"
        else:
            raise ValueError(
                "No API key found. Set one of:\n"
                "  - OPENAI_API_KEY (for OpenAI API)\n"
                "  - GITHUB_TOKEN (for GitHub Copilot API)\n"
                "in your .env file"
            )

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
                        # Try to extract JSON from response
                        json_str = content
                        if "```json" in content:
                            json_str = content.split("```json")[1].split("```")[0]
                        elif "```" in content:
                            json_str = content.split("```")[1].split("```")[0]

                        parsed = json.loads(json_str.strip())
                        return True, {
                            "match": parsed.get("match", False),
                            "score": parsed.get("score", 0),
                            "rationale": parsed.get("rationale", "")
                        }
                    except json.JSONDecodeError:
                        # Try to extract info from non-JSON response
                        return False, {"error": f"Invalid JSON response: {content[:200]}"}

                elif response.status_code == 429:
                    # Rate limited - wait and retry
                    wait_time = retry_delay * (2 ** attempt)
                    print(f"Rate limited. Waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue

                elif response.status_code == 401:
                    return False, {"error": f"Authentication failed. Check your API key."}

                else:
                    return False, {"error": f"API error {response.status_code}: {response.text[:200]}"}

            except requests.exceptions.Timeout:
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                return False, {"error": "Request timeout"}

            except Exception as e:
                return False, {"error": str(e)}

        return False, {"error": "Max retries exceeded"}

    def test_connection(self) -> bool:
        """
        Test API connection with a simple request.

        Returns:
            True if connection successful, False otherwise
        """
        try:
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": "Say 'OK' if you can read this."}],
                "max_tokens": 10
            }
            response = requests.post(
                self.api_endpoint,
                headers=self.headers,
                json=payload,
                timeout=30
            )
            if response.status_code == 200:
                print("Connection test: SUCCESS")
                return True
            else:
                print(f"Connection test: FAILED ({response.status_code})")
                return False
        except Exception as e:
            print(f"Connection test: FAILED ({e})")
            return False

    def list_models(self) -> list:
        """
        List available models (OpenAI only).

        Returns:
            List of model names
        """
        if self.backend != "openai":
            return self.BACKENDS[self.backend]["models"]

        models_endpoint = "https://api.openai.com/v1/models"

        try:
            response = requests.get(models_endpoint, headers=self.headers, timeout=30)
            if response.status_code == 200:
                data = response.json().get("data", [])
                gpt_models = [m["id"] for m in data if "gpt" in m["id"]]
                return sorted(gpt_models)
        except Exception as e:
            print(f"Error listing models: {e}")

        return self.BACKENDS[self.backend]["models"]
