import importlib.util
from pathlib import Path
import pytest

MODULES = [
    (
        "character_generator",
        Path("Character-Generator/app.py"),
        ["gradio", "torch", "diffusers", "datasets", "pydantic"],
    ),
    (
        "flux_lora_dlc",
        Path("FLUX-LoRA-DLC/app.py"),
        ["gradio", "torch", "diffusers", "numpy", "PIL"],
    ),
    (
        "faceswap",
        Path("faceswap/app.py"),
        ["gradio", "torch", "cv2", "insightface", "onnxruntime", "moviepy.editor", "numpy"],
    ),
    (
        "self_forcing",
        Path("self-forcing/app.py"),
        ["gradio", "torch", "huggingface_hub", "omegaconf", "transformers", "numpy", "imageio", "av"],
    ),
]

@pytest.mark.parametrize("name,path,deps", MODULES)
def test_create_app(name: str, path: Path, deps: list[str]) -> None:
    """Import module and execute its `create_app` function if dependencies are available."""
    for dep in deps:
        pytest.importorskip(dep)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)  # type: ignore[arg-type]
    app = module.create_app()
    assert app is not None
