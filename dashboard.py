import gradio as gr
import importlib.util
import os

from utils.ollama_client import AVAILABLE_MODELS, set_model, get_model


def load_create_app(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.create_app

character_create_app = load_create_app('character_generator', os.path.join('Character-Generator', 'app.py'))
flux_create_app = load_create_app('flux_lora_dlc', os.path.join('FLUX-LoRA-DLC', 'app.py'))
faceswap_create_app = load_create_app('faceswap', os.path.join('faceswap', 'app.py'))
selfforcing_create_app = load_create_app('self_forcing', os.path.join('self-forcing', 'app.py'))


def create_settings_app():
    """Settings tab with model selection."""
    with gr.Blocks() as app:
        gr.Markdown("## Settings")
        model_dropdown = gr.Dropdown(
            choices=AVAILABLE_MODELS,
            value=get_model(),
            label="Ollama Model",
        )

        model_dropdown.change(set_model, inputs=model_dropdown, outputs=None)

    return app

apps = [
    character_create_app(),
    flux_create_app(),
    faceswap_create_app(),
    selfforcing_create_app(),
    create_settings_app(),
]
labels = [
    'Character Generator',
    'FLUX LoRA DLC',
    'FaceSwap',
    'Self-Forcing',
    'Settings',
]

demo = gr.TabbedInterface(apps, labels)

if __name__ == '__main__':
    demo.queue().launch()

