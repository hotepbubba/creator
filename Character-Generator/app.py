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
    
            examples = gr.Examples(
                [
                    "In a world where magic is real and dragons roam the skies, a group of adventurers set out to find the legendary sword of the dragon king.",
                    "Welcome to Aethoria, a vast and mysterious realm where the laws of physics bend to the will of ancient magic. This world is comprised of countless floating islands suspended in an endless sky, each one a unique ecosystem teeming with life and secrets. The islands of Aethoria range from lush, verdant jungles to barren, crystalline deserts. Some are no larger than a city block, while others span hundreds of miles. Connecting these disparate landmasses are shimmering bridges of pure energy, and those brave enough to venture off the beaten path can find hidden portals that instantly transport them across great distances. Aethoria's inhabitants are as diverse as its landscapes. Humans coexist with ethereal beings of light, rock-skinned giants, and shapeshifting creatures that defy classification. Ancient ruins dot the islands, hinting at long-lost civilizations and forgotten technologies that blur the line between science and sorcery. The world is powered by Aether, a mystical substance that flows through everything. Those who can harness its power become formidable mages, capable of manipulating reality itself. However, Aether is a finite resource, and its scarcity has led to conflicts between the various factions vying for control. In the skies between the islands, magnificent airships sail on currents of magic, facilitating trade and exploration. Pirates and sky raiders lurk in the cloudy depths, always on the lookout for unsuspecting prey. Deep beneath the floating lands lies the Undervoid, a dark and treacherous realm filled with nightmarish creatures and untold riches. Only the bravest adventurers dare to plumb its depths, and fewer still return to tell the tale. As an ever-present threat, the Chaos Storms rage at the edges of the known world, threatening to consume everything in their path. It falls to the heroes of Aethoria to uncover the secrets of their world and find a way to push back the encroaching darkness before it's too late. In Aethoria, every island holds a story, every creature has a secret, and every adventure could change the fate of this wondrous, imperiled world.",
                    "In a world from my imagination, there is a city called 'Orakis'. floating in the sky on pillars of pure light. The walls of the city are made of crystal glass, constantly reflecting the colors of dawn and dusk, giving it an eternal celestial glow. The buildings breathe and change their shapes according to the seasons—they grow in spring, strengthen in summer, and begin to fade in autumn until they become mist in winter.",
                ],
                world_description,
            )
    
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
