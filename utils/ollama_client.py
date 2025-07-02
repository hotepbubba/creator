import os
import requests

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
_current_model = None

def set_model(name: str):
    """Set the default model name used for generation."""
    global _current_model
    _current_model = name


def generate(prompt: str, model: str | None = None) -> str:
    """Generate text from the Ollama API."""
    m = model or _current_model
    if not m:
        raise ValueError("model must be provided or set via set_model()")
    url = f"{OLLAMA_HOST}/api/generate"
    resp = requests.post(url, json={"model": m, "prompt": prompt, "stream": False})
    resp.raise_for_status()
    data = resp.json()
    return data.get("response", "")
