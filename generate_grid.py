import datetime;
from diffusers import LMSDiscreteScheduler, DDIMScheduler
import json
import numpy as np
import torch

from diffusiontools.tiling import StableDiffusionTilingPipeline

### CONFIG START
n = 1
#sche = "ddim"
sche = "lms"
gc = 8
seed = 4821955176
steps = 100
#steps = 50
#steps = 10
#steps = 1

suffix = "elegant, highly detailed, smooth, sharp focus, artstation, stunning masterpiece"
filler = f"a pathways and staircases floating in deep space, a constellation of stars and planets and black holes in the background, art by style of escher and patrick mcenvoy and michael komarck, {suffix}"
prompt = [
    [
        f"{filler}",
        f"An archway leading to a starry nigh, art by van gogh and patrick mcenvoy and michael komarck, {suffix}",
        f"{filler}",
        f"An archway leading to a nightmare world full of bloody eyes and jaws, art by Hans Ruedi Giger and patrick mcenvoy and michael komarck, {suffix}",
        f"{filler}",
    ],
    [
        #"An archway leading to a landscape of hell, art by Hieronymus Bosch, {suffix}",
        #"A rift in space showing a view of a massive tower of babel in the center of hell, in the style of Hieronymus Bosch, a constellation of stars and planets and black holes in the background, {suffix}",
        "A view from a cave of a landscape of hell, with lost souls wandering about, art by patrick mcenvoy and michael komarck and greg rutkowski, {suffix}",
        f"{filler}",
        "A hooded figure facing the camera, with tentacles instead of feet, holding a large golden intricate key, art by patrick mcenvoy and michael komarck, {suffix}",
        f"{filler}",
        f"A An archway leading to a meadow with dragons with rainbow crystal mountains and forests in the background, art by alayna danner, {suffix}"
    ]
]

gc_tiles = [
    [None, 8.0, None, None, None],
    [None, None, 8.0, None, 6.0],
]

tile_height = 640
tile_width = 640
tile_row_overlap = 256
tile_col_overlap = 256
cpu_vae = True

# suffix = "elegant, highly detailed, smooth, sharp focus, artstation, stunning masterpiece"
# filler = f"a pathways and staircases floating in deep space, a constellation of stars and planets and black holes in the background, art by style of escher and patrick mcenvoy and michael komarck, {suffix}"
# prompt = [
#     [
#         f"{filler}",
#         f"An archway leading to a starry nigh, art by van gogh and patrick mcenvoy and michael komarck, {suffix}",
#         f"{filler}",
#         f"An archway leading to a nightmare world full of bloody eyes and jaws, art by Hans Ruedi Giger and patrick mcenvoy and michael komarck, {suffix}",
#         f"{filler}",
#     ],
#     [
#         "A An archway leading to a gothic castle under a crimson moon, art by Ayami Kojima in the style of bloodborne, {suffix}",
#         f"{filler}",
#         "A hooded figure facing the camera, with tentacles instead of feet, holding a large golden intricate key, art by patrick mcenvoy and michael komarck, {suffix}",
#         f"{filler}",
#         f"A An archway leading to a meadow with dragons with rainbow crystal mountains and forests in the background, art by alayna danner, {suffix}"
#     ]
# ]

# gc_tiles = [
#     [None, 9.0, None, None, None],
#     [None, None, 8.0, None, 6.0],
# ]

# tile_height = 640
# tile_width = 640
# tile_row_overlap = 256
# tile_col_overlap = 256
# cpu_vae = True

# forest = "a forest"
# suffix = "intricate, elegant, highly detailed, smooth, sharp focus, artstation, stunning masterpiece, impressive colors"
# prompt = [
#     [
#         f"{forest}, ukiyo-e, {suffix}",
#         f"{forest}, ukiyo-e, {suffix}",
#         f"{forest}, by velazquez, {suffix}",
#         f"{forest}, by velazquez, {suffix}",
#         f"{forest}, impressionist style by van gogh, {suffix}",
#         f"{forest}, impressionist style by van gogh, {suffix}",
#         f"{forest}, cubist style by Pablo Picasso {suffix}",
#         f"{forest}, cubist style by Pablo Picasso {suffix}",
#         f"{forest}, 80s synthwave style, {suffix}",
#         f"{forest}, 80s synthwave style, {suffix}",
#     ],
# ]

# forest = "a forest"
# suffix = "intricate, elegant, highly detailed, smooth, sharp focus, artstation, stunning masterpiece, impressive colors"
# prompt = [
#     [
#         f"{forest}, in the style of a prehistoric cave painting, {suffix}",
#         f"{forest}, int the style of an ancient egyptian painting, {suffix}",
#         f"{forest}, ukiyo-e, {suffix}",
#         f"{forest}, in the style of a middle ages painting, {suffix}",
#         f"{forest}, by velazquez, {suffix}",
#         f"{forest}, by van gogh, {suffix}",
#         f"{forest}, by Gustav Klimt, {suffix}",
#         f"{forest}, pop-art by Roy Lichstenstein, {suffix}",
#         f"{forest}, cubist style by Pablo Picasso {suffix}",
#         f"{forest}, 80s synthwave style, {suffix}",
#         f"{forest}, pixel-art, {suffix}",
#         f"{forest}, in the style of minecraft, {suffix}",
#         f"{forest}, unreal engine render 8K in the style of pixar, {suffix}",
#         f"{forest}, cyberpunk futuristic half-robotic style, {suffix}"
#     ],
# ]

# prompt = [
#     [
#         f"{forest}, by salvador dali, {suffix}",
#         f"{frame}, synthwave style, {suffix}",
#         f"{forest}, art by escher, {suffix}",
#         f"{frame}, sketch by picasso, {suffix}",
#         f"{forest}, by leonid afremov, {suffix}",
#         f"{frame}, in the style of super mario world, {suffix}",
#         f"{forest}, ukiyo-e, {suffix}",
#         f"{forest}, art by hieronymus bosch, {suffix}"
#     ],
#     [
#         f"{frame}, drawn on a blackboard, {suffix}",
#         f"{forest}, by joan miro, {suffix}",
#         f"{forest}, art by van gogh, {suffix}",
#         f"{forest}, by monet, {suffix}",
#         f"{frame}, art by roy lichtenstein, {suffix}",
#         f"{forest}, by ghibli studio, {suffix}",
#         f"{forest}, art by gustav klimt, {suffix}",
#         f"{frame}, street art by banksy, {suffix}"
#     ]
#         # f"a forest, in the style of monogatari, {suffix}",
#         # f"a forest, art by goya, {suffix}",
#         # f"a forest, in the style of super mario world, {suffix}",
#         # f"a forest, relief on stone, {suffix}",
#         # f"a forest, steampunk, {suffix}",
# ]
# gc_tiles = None

# tile_height = 768
# tile_width = 512
# tile_row_overlap = 256
# tile_col_overlap = 256
# cpu_vae = True

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
    pipeargs = {
        "guidance_scale": gc_image,
        "num_inference_steps": steps_image,
        "generator": torch.Generator("cuda").manual_seed(seed_image),
        "prompt": prompt,
        "tile_height": tile_height, 
        "tile_width": tile_width, 
        "tile_row_overlap": tile_row_overlap, 
        "tile_col_overlap": tile_col_overlap,
        "guidance_scale_tiles": gc_tiles,
        "cpu_vae": cpu_vae,
    }
    image = pipe(**pipeargs)["sample"][0]
    ct = datetime.datetime.now()
    outname = f"{ct}_{prompt[0][0][0:100]}_{tile_height}x{tile_width}_sche{sche}_seed{seed_image}_gc{gc_image}_steps{steps_image}"
    image.save(f"outputs/{outname}.png")
    with open(f"logs/{outname}.txt", "w") as f:
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
                "cpu_vae": cpu_vae,
            },
            f,
            sort_keys=True,
            indent=4
        )
