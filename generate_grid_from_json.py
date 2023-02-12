import argparse
import datetime
from diffusers import LMSDiscreteScheduler, DDIMScheduler
import json
from pathlib import Path
import torch

from mixdiff.tiling import StableDiffusionTilingPipeline

def generate_grid(generation_arguments):
    model_id = "CompVis/stable-diffusion-v1-4"
    # Prepared scheduler
    if generation_arguments["scheduler"] == "ddim":
        scheduler = DDIMScheduler()
    elif generation_arguments["scheduler"] == "lms":
        scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)
    else:
        raise ValueError(f"Unrecognized scheduler {generation_arguments['scheduler']}")
    pipe = StableDiffusionTilingPipeline.from_pretrained(model_id, scheduler=scheduler, use_auth_token=True).to("cuda:0")

    pipeargs = {
        "guidance_scale": generation_arguments["gc"],
        "num_inference_steps": generation_arguments["steps"],
        "seed": generation_arguments["seed"],
        "prompt": generation_arguments["prompt"],
        "tile_height": generation_arguments["tile_height"], 
        "tile_width": generation_arguments["tile_width"], 
        "tile_row_overlap": generation_arguments["tile_row_overlap"], 
        "tile_col_overlap": generation_arguments["tile_col_overlap"],
        "guidance_scale_tiles": generation_arguments["gc_tiles"],
        "cpu_vae": generation_arguments["cpu_vae"] if "cpu_vae" in generation_arguments else False,
    }
    if "seed_tiles" in generation_arguments: pipeargs = {**pipeargs, "seed_tiles": generation_arguments["seed_tiles"]}
    if "seed_tiles_mode" in generation_arguments: pipeargs = {**pipeargs, "seed_tiles_mode": generation_arguments["seed_tiles_mode"]}
    if "seed_reroll_regions" in generation_arguments: pipeargs = {**pipeargs, "seed_reroll_regions": generation_arguments["seed_reroll_regions"]}
    image = pipe(**pipeargs)["sample"][0]
    outname = "output"
    outpath = "./outputs"
    Path(outpath).mkdir(parents=True, exist_ok=True)
    image.save(f"{outpath}/{outname}.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate a stable diffusion grid using a JSON file with all configuration parameters.')
    parser.add_argument('config', type=str, help='Path to configuration file')
    args = parser.parse_args()
    with open(args.config, "r") as f:
        generation_arguments = json.load(f)
    generate_grid(generation_arguments)
