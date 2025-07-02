import gradio as gr
import importlib.util
import os


def load_create_app(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.create_app

character_create_app = load_create_app('character_generator', os.path.join('Character-Generator', 'app.py'))
flux_create_app = load_create_app('flux_lora_dlc', os.path.join('FLUX-LoRA-DLC', 'app.py'))
faceswap_create_app = load_create_app('faceswap', os.path.join('faceswap', 'app.py'))
selfforcing_create_app = load_create_app('self_forcing', os.path.join('self-forcing', 'app.py'))

apps = [
    character_create_app(),
    flux_create_app(),
    faceswap_create_app(),
    selfforcing_create_app(),
]
labels = [
    'Character Generator',
    'FLUX LoRA DLC',
    'FaceSwap',
    'Self-Forcing',
]

demo = gr.TabbedInterface(apps, labels)

if __name__ == '__main__':
    demo.queue().launch()

