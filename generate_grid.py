from copy import deepcopy
import datetime
from diffusers import LMSDiscreteScheduler, DDIMScheduler
import git
import json
import numpy as np
from pathlib import Path

from mixdiff import StableDiffusionTilingPipeline

### CONFIG START
n = 3
sche = "lms"
gc = 8
seed = None
steps = 50

suffix = "elegant, highly detailed, smooth, sharp focus, artstation, stunning masterpiece"
prompt = [
    [
        f"A charming house in the countryside, by jakub rozalski, sunset lighting, {suffix}",
        f"A dirt road in the countryside crossing pastures, by jakub rozalski, sunset lighting, {suffix}",
        f"An old and rusty giant robot lying on a dirt road, by jakub rozalski, dark sunset lighting, {suffix}"
    ]
]

gc_tiles = [
    [None, None, None],
]

seed_tiles = [
    [None, None, None],
]
seed_tiles_mode = StableDiffusionTilingPipeline.SeedTilesMode.FULL.value

seed_reroll_regions = []

tile_height = 640
tile_width = 640
tile_row_overlap = 256
tile_col_overlap = 256
cpu_vae = False

### CONFIG END

# Prepared scheduler
if sche == "ddim":
    scheduler = DDIMScheduler()
elif sche == "lms":
    scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)
else:
    raise ValueError(f"Unrecognized scheduler {sche}")

# Load model
model_id = "CompVis/stable-diffusion-v1-4"
pipelinekind = StableDiffusionTilingPipeline
pipe = pipelinekind.from_pretrained(model_id, scheduler=scheduler, use_auth_token=True).to("cuda:0")

for _ in range(n):
    # Prepare parameters
    gc_image = gc if gc is not None else np.random.randint(5, 30)
    steps_image = steps if steps is not None else np.random.randint(50, 150)
    seed_image = seed if seed is not None else np.random.randint(9999999999)
    seed_tiles_image = deepcopy(seed_tiles)
    for row in range(len(seed_tiles)):
        for col in range(len(seed_tiles[0])):
            if seed_tiles[row][col] == "RNG":
                seed_tiles_image[row][col] = np.random.randint(9999999999)
    seed_reroll_regions_image = deepcopy(seed_reroll_regions)
    for i in range(len(seed_reroll_regions)):
        row_init, row_end, col_init, col_end, seed_reroll = seed_reroll_regions[i]
        if seed_reroll == "RNG":
            seed_reroll = np.random.randint(9999999999)
        seed_reroll_regions_image[i] = (row_init, row_end, col_init, col_end, seed_reroll)
    pipeargs = {
        "guidance_scale": gc_image,
        "num_inference_steps": steps_image,
        "seed": seed_image,
        "prompt": prompt,
        "tile_height": tile_height, 
        "tile_width": tile_width, 
        "tile_row_overlap": tile_row_overlap, 
        "tile_col_overlap": tile_col_overlap,
        "guidance_scale_tiles": gc_tiles,
        "seed_tiles": seed_tiles_image,
        "seed_tiles_mode": seed_tiles_mode,
        "seed_reroll_regions": seed_reroll_regions_image,
        "cpu_vae": cpu_vae,
    }
    image = pipe(**pipeargs)["sample"][0]
    ct = datetime.datetime.now()
    outname = f"{ct}_{prompt[0][0][0:100]}_{tile_height}x{tile_width}_sche{sche}_seed{seed_image}_gc{gc_image}_steps{steps_image}"
    outpath = "./outputs"
    Path(outpath).mkdir(parents=True, exist_ok=True)
    image.save(f"{outpath}/{outname}.png")
    logspath = "./logs"
    Path(logspath).mkdir(parents=True, exist_ok=True)
    with open(f"{logspath}/{outname}.json", "w") as f:
        json.dump(
            {
                "prompt": prompt,
                "tile_height": tile_height,
                "tile_width": tile_width,
                "tile_row_overlap": tile_row_overlap,
                "tile_col_overlap": tile_col_overlap,
                "scheduler": sche,
                "seed": seed_image,
                "gc": gc_image,
                "gc_tiles": gc_tiles,
                "steps": steps_image,
                "seed_tiles": seed_tiles_image,
                "seed_tiles_mode": seed_tiles_mode,
                "seed_reroll_regions": seed_reroll_regions_image,
                "cpu_vae": cpu_vae,
                "git_commit": git.Repo(search_parent_directories=True).head.object.hexsha,
            },
            f,
            sort_keys=True,
            indent=4
        )
