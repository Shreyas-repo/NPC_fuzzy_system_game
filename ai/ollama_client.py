"""
Shared Ollama API client for generation, embeddings, and lightweight analysis.
"""
import json
from typing import Dict, List, Optional

import requests


class OllamaClient:
    """Thin wrapper around Ollama HTTP APIs."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        chat_model: str = "deepseek-r1:7b",
        embedding_model: str = "nomic-embed-text",
        analysis_model: Optional[str] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.chat_model = chat_model
        self.embedding_model = embedding_model
        self.analysis_model = analysis_model or chat_model

    def is_available(self) -> bool:
        """Return True if Ollama responds and models can be queried."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=2)
            return response.status_code == 200
        except Exception:
            return False

    def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        options: Optional[Dict] = None,
        timeout: float = 45,
    ) -> str:
        """Generate text from a prompt."""
        payload = {
            "model": model or self.chat_model,
            "prompt": prompt,
            "stream": False,
        }
        if options:
            payload["options"] = options

        response = requests.post(f"{self.base_url}/api/generate", json=payload, timeout=timeout)
        response.raise_for_status()
        return response.json().get("response", "").strip()

    def embed(self, text: str, model: Optional[str] = None) -> List[float]:
        """Create an embedding vector for text."""
        payload = {
            "model": model or self.embedding_model,
            "prompt": text,
        }
        response = requests.post(f"{self.base_url}/api/embeddings", json=payload, timeout=45)
        response.raise_for_status()
        data = response.json()
        vec = data.get("embedding", [])
        return vec if isinstance(vec, list) else []

    def analyze_text(self, text: str) -> Optional[Dict]:
        """
        Ask Ollama to return strict JSON sentiment+emotion analysis.
        Returns None on parse/runtime failures.
        """
        prompt = (
            "Analyze this text and return ONLY valid JSON with keys: "
            "sentiment (-1 to 1), joy, anger, fear, sadness, trust, surprise, disgust, curiosity (0 to 1).\n"
            f"Text: {text}"
        )
        try:
            raw = self.generate(
                prompt,
                model=self.analysis_model,
                options={"temperature": 0.1, "num_predict": 120},
            )
            start = raw.find("{")
            end = raw.rfind("}")
            if start == -1 or end == -1 or end <= start:
                return None
            parsed = json.loads(raw[start:end + 1])
            return parsed if isinstance(parsed, dict) else None
        except Exception:
            return None
