import os
import requests

__all__ = [
    "AVAILABLE_MODELS",
    "set_model",
    "get_model",
    "generate",
]

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

# List of models available to all apps using this client
AVAILABLE_MODELS = [
    "llama3",
    "mistral",
]

# Currently selected model
_current_model = os.environ.get("OLLAMA_MODEL", AVAILABLE_MODELS[0])

def set_model(name: str):
    """Set the default model name used for generation."""
    global _current_model
    _current_model = name


def get_model() -> str:
    """Return the currently selected model name."""
    return _current_model


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
