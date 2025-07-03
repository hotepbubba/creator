from datasets import load_dataset
import gradio as gr
import json
import torch
from diffusers import FluxPipeline, AutoencoderKL
from live_preview_helpers import flux_pipe_call_that_returns_an_iterable_of_images
from pydantic import BaseModel
from utils.ollama_client import generate
from utils.cache import get_cache_dir

__all__ = ["create_app"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class CharacterDescription(BaseModel):
    name: str
    background: str
    appearance: str
    personality: str
    skills_and_abilities: str
    goals: str
    conflicts: str
    backstory: str
    current_situation: str
    spoken_lines: list[str]

pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.bfloat16,
    cache_dir=get_cache_dir(),
).to(device)
good_vae = AutoencoderKL.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    subfolder="vae",
    torch_dtype=torch.bfloat16,
    cache_dir=get_cache_dir(),
).to(device)
# pipe.enable_sequential_cpu_offload()
# pipe.vae.enable_slicing()
# pipe.vae.enable_tiling()
# pipe.to(torch.float16)
pipe.flux_pipe_call_that_returns_an_iterable_of_images = flux_pipe_call_that_returns_an_iterable_of_images.__get__(pipe)

ds = load_dataset("MohamedRashad/FinePersonas-Lite", split="train")

prompt_template = """Generate a character with this persona description:
{persona_description}
---
In a world with this description:
{world_description}
"""

world_description_prompt = "Generate a unique and random world description (Don't Write anything else except the world description)."

def get_random_world_description():
    result = generate(world_description_prompt)
    if "</think>" in result:
        result = result[result.index("</think>")+len("</think>"):].strip()
    return result

def get_random_persona_description():
    return ds.shuffle().select([100])[0]["persona"]

def infer_flux(character_json):
    for image in pipe.flux_pipe_call_that_returns_an_iterable_of_images(
        prompt=character_json["appearance"],
        guidance_scale=3.5,
        num_inference_steps=28,
        width=1024,
        height=1024,
        generator=torch.Generator("cpu").manual_seed(0),
        output_type="pil",
        good_vae=good_vae,
    ):
        yield image

def generate_character(world_description, persona_description, progress=gr.Progress(track_tqdm=True)):
    prompt_content = prompt_template.format(
        persona_description=persona_description, 
        world_description=world_description
    )
    result = generate(prompt_content)
    return json.loads(result)

app_description = """
- This app generates a character in JSON format based on a persona description and a world description.
- The character's appearance is generated using [FLUX-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) and the character description is generated using Google Gemini 2.5 Flash.
- The persona description is randomly selected from the [FinePersonas-Lite](https://huggingface.co/datasets/MohamedRashad/FinePersonas-Lite) dataset.

**Note:** I recommend starting with the world description (you can write one or loop over randomly generated ones) and then try different persona descriptions to generate interesting characters for the world you created.
"""

def create_app():
    with gr.Blocks(title="Character Generator") as app:
        with gr.Column():
            gr.HTML("<center><h1>Character Generator</h1></center>")
            gr.Markdown(app_description.strip())
            with gr.Column():
                with gr.Row():
                    world_description = gr.Textbox(lines=10, label="World Description", scale=4)
                    persona_description = gr.Textbox(lines=10, label="Persona Description", value=get_random_persona_description(), scale=1)
                with gr.Row(equal_height=True):
                    random_world_button = gr.Button(value="Get Random World Description", variant="secondary", scale=1)
                    submit_button = gr.Button(value="Generate Interesting Character!", variant="primary", scale=5)
                    random_persona_button = gr.Button(value="Get Random Persona Description", variant="secondary", scale=1)
            with gr.Row(equal_height=True):
                character_image = gr.Image(label="Character Image", height=1024, width=1024)
                character_json = gr.JSON(label="Character Description")
    
        submit_button.click(
            generate_character, [world_description, persona_description], outputs=[character_json]
        ).then(fn=infer_flux, inputs=[character_json], outputs=[character_image])
        random_world_button.click(
            get_random_world_description, outputs=[world_description]
        )
        random_persona_button.click(
            get_random_persona_description, outputs=[persona_description]
        )
    return app


if __name__ == "__main__":
    create_app().queue().launch(share=False)
