# Mixture of Diffusers

![2022-10-12 15_35_27 305133_A charming house in the countryside, by jakub rozalski, sunset lighting, elegant, highly detailed, s_640x640_schelms_seed7178915308_gc8_steps50](https://user-images.githubusercontent.com/9654655/195362341-bc7766c2-f5c6-40f2-b457-59277aa11027.png)

[![Unit tests](https://github.com/albarji/mixture-of-diffusers/actions/workflows/python-tests.yml/badge.svg)](https://github.com/albarji/mixture-of-diffusers/actions/workflows/python-tests.yml)

[![huggingface space](https://camo.githubusercontent.com/00380c35e60d6b04be65d3d94a58332be5cc93779f630bcdfc18ab9a3a7d3388/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f25463025394625413425393725323048756767696e67253230466163652d5370616365732d626c7565)](https://huggingface.co/spaces/albarji/mixture-of-diffusers)

This repository holds various scripts and tools implementing a method for integrating a mixture of different diffusion processes collaborating to generate a single image. Each diffuser focuses on a particular region on the image, taking into account boundary effects to promote a smooth blending.

If you prefer a more user friendly graphical interface to use this algorithm, I recommend trying the [Tiled Diffusion & VAE](https://github.com/pkuliyi2015/multidiffusion-upscaler-for-automatic1111) plugin developed by pkuliyi2015 for [AUTOMATIC1111's stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui).

## Motivation

Current image generation methods, such as Stable Diffusion, struggle to position objects at specific locations. While the content of the generated image (somewhat) reflects the objects present in the prompt, it is difficult to frame the prompt in a way that creates an specific composition. For instance, take a prompt expressing a complex composition such as

> A charming house in the countryside on the left,
> in the center a dirt road in the countryside crossing pastures,
> on the right an old and rusty giant robot lying on a dirt road,
> by jakub rozalski,
> sunset lighting on the left and center, dark sunset lighting on the right
> elegant, highly detailed, smooth, sharp focus, artstation, stunning masterpiece

Out of a sample of 20 Stable Diffusion generations with different seeds, the generated images that align best with the prompt are the following:

<table>
  <tr>
    <td><img src="https://user-images.githubusercontent.com/9654655/195373001-ad23b7c4-f5b1-4e5b-9aa1-294441ed19ed.png" width="300"></td>
    <td><img src="https://user-images.githubusercontent.com/9654655/195373174-8d85dd96-310e-48fa-b112-d9902685f22e.png" width="300"></td>
    <td><img src="https://user-images.githubusercontent.com/9654655/195373200-59eeec1e-e1b8-464d-b72e-e28a9004d269.png" width="300"></td>
  </tr>
</table>

The method proposed here strives to provide a better tool for image composition by using several diffusion processes in parallel, each configured with a specific prompt and settings, and focused on a particular region of the image. For example, the following are three outputs from this method, using the following prompts from left to right:

* "**A charming house in the countryside, by jakub rozalski, sunset lighting**, elegant, highly detailed, smooth, sharp focus, artstation, stunning masterpiece"
* "**A dirt road in the countryside crossing pastures, by jakub rozalski, sunset lighting**, elegant, highly detailed, smooth, sharp focus, artstation, stunning masterpiece"
* "**An old and rusty giant robot lying on a dirt road, by jakub rozalski, dark sunset lighting**, elegant, highly detailed, smooth, sharp focus, artstation, stunning masterpiece"

![2022-10-12 15_25_40 021063_A charming house in the countryside, by jakub rozalski, sunset lighting, elegant, highly detailed, s_640x640_schelms_seed9764851938_gc8_steps50](https://user-images.githubusercontent.com/9654655/195362152-6f3af44d-cf8a-494b-8cf8-36acd8f86871.png)
![2022-10-12 15_32_11 563087_A charming house in the countryside, by jakub rozalski, sunset lighting, elegant, highly detailed, s_640x640_schelms_seed2096547054_gc8_steps50](https://user-images.githubusercontent.com/9654655/195362315-8c2d01a8-62f2-4d96-90ca-9ad22f69398e.png)
![2022-10-12 15_35_27 305133_A charming house in the countryside, by jakub rozalski, sunset lighting, elegant, highly detailed, s_640x640_schelms_seed7178915308_gc8_steps50](https://user-images.githubusercontent.com/9654655/195362341-bc7766c2-f5c6-40f2-b457-59277aa11027.png)

The mixture of diffusion processes is done in a way that harmonizes the generation process, preventing "seam" effects in the generated image.

Using several diffusion processes in parallel has also practical advantages when generating very large images, as the GPU memory requirements are similar to that of generating an image of the size of a single tile.

## Usage

This repository provides two new pipelines, `StableDiffusionTilingPipeline` and `StableDiffusionCanvasPipeline`, that extend the standard Stable Diffusion pipeline from [Diffusers](https://github.com/huggingface/diffusers). They feature new options that allow defining the mixture of diffusers, which are distributed as a number of "diffusion regions" over the image to be generated. `StableDiffusionTilingPipeline` is simpler to use and arranges the diffusion regions as a grid over the canvas, while `StableDiffusionCanvasPipeline` allows a more flexible placement and also features image2image capabilities.

### Prerequisites

Since this work is based on Stable Diffusion models, you will need to [request access and accept the usage terms of Stable Diffusion](https://huggingface.co/CompVis/stable-diffusion#model-access). You will also need to [configure your Hugging Face User Access Token](https://huggingface.co/docs/hub/security-tokens) in your running environment.

### StableDiffusionTilingPipeline

The header image in this repo can be generated as follows

```python
from diffusers import LMSDiscreteScheduler
from mixdiff import StableDiffusionTilingPipeline

# Creater scheduler and model (similar to StableDiffusionPipeline)
scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)
pipeline = StableDiffusionTilingPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", scheduler=scheduler, use_auth_token=True).to("cuda:0")

# Mixture of Diffusers generation
image = pipeline(
    prompt=[[
        "A charming house in the countryside, by jakub rozalski, sunset lighting, elegant, highly detailed, smooth, sharp focus, artstation, stunning masterpiece",
        "A dirt road in the countryside crossing pastures, by jakub rozalski, sunset lighting, elegant, highly detailed, smooth, sharp focus, artstation, stunning masterpiece",
        "An old and rusty giant robot lying on a dirt road, by jakub rozalski, dark sunset lighting, elegant, highly detailed, smooth, sharp focus, artstation, stunning masterpiece"
    ]],
    tile_height=640,
    tile_width=640,
    tile_row_overlap=0,
    tile_col_overlap=256,
    guidance_scale=8,
    seed=7178915308,
    num_inference_steps=50,
)["sample"][0]
```

The prompts must be provided as a list of lists, where each list represents a row of diffusion regions. The geometry of the canvas is inferred from these lists, e.g. in the example above we are creating a grid of 1x3 diffusion regions (1 row and 3 columns). The rest of parameters provide information on the size of these regions, and how much they overlap with their neighbors.

Alternatively, it is possible to specify the grid parameters through a JSON configuration file. In the following example a grid of 10x1 tiles is configured to generate a forest in changing styles:

![gridExampleLabeled](https://user-images.githubusercontent.com/9654655/195371664-54d8a599-25d8-46ba-b823-3c7726ecb6ff.png)

A `StableDiffusionTilingPipeline` is configured to use 10 prompts with changing styles. Each tile takes a shape of 768x512 pixels, and tiles overlap 256 pixels to avoid seam effects. All the details are specified in a configuration file:

```json
{
    "cpu_vae": true,
    "gc": 8,
    "gc_tiles": null,
    "prompt": [
        [
            "a forest, ukiyo-e, intricate, elegant, highly detailed, smooth, sharp focus, artstation, stunning masterpiece, impressive colors",
            "a forest, ukiyo-e, intricate, elegant, highly detailed, smooth, sharp focus, artstation, stunning masterpiece, impressive colors",
            "a forest, by velazquez, intricate, elegant, highly detailed, smooth, sharp focus, artstation, stunning masterpiece, impressive colors",
            "a forest, by velazquez, intricate, elegant, highly detailed, smooth, sharp focus, artstation, stunning masterpiece, impressive colors",
            "a forest, impressionist style by van gogh, intricate, elegant, highly detailed, smooth, sharp focus, artstation, stunning masterpiece, impressive colors",
            "a forest, impressionist style by van gogh, intricate, elegant, highly detailed, smooth, sharp focus, artstation, stunning masterpiece, impressive colors",
            "a forest, cubist style by Pablo Picasso intricate, elegant, highly detailed, smooth, sharp focus, artstation, stunning masterpiece, impressive colors",
            "a forest, cubist style by Pablo Picasso intricate, elegant, highly detailed, smooth, sharp focus, artstation, stunning masterpiece, impressive colors",
            "a forest, 80s synthwave style, intricate, elegant, highly detailed, smooth, sharp focus, artstation, stunning masterpiece, impressive colors",
            "a forest, 80s synthwave style, intricate, elegant, highly detailed, smooth, sharp focus, artstation, stunning masterpiece, impressive colors"
        ]
    ],
    "scheduler": "lms",
    "seed": 639688656,
    "steps": 50,
    "tile_col_overlap": 256,
    "tile_height": 768,
    "tile_row_overlap": 256,
    "tile_width": 512
}
```

You can try generating this image using this configuration file by running

```bash
python generate_grid_from_json.py examples/linearForest.json
```

The full list of arguments to a `StableDiffusionTilingPipeline` is:

> `prompt`: either a single string (no tiling) or a list of lists with all the prompts to use (one list for each row of tiles). This will also define the tiling structure.
>
> `num_inference_steps`: number of diffusions steps.
>
> `guidance_scale`: classifier-free guidance.
>
> `seed`: general random seed to initialize latents.
>
> `tile_height`: height in pixels of each grid tile.
>
> `tile_width`: width in pixels of each grid tile.
>
> `tile_row_overlap`: number of overlap pixels between tiles in consecutive rows.
>
> `tile_col_overlap`: number of overlap pixels between tiles in consecutive columns.
>
> `guidance_scale_tiles`: specific weights for classifier-free guidance in each tile.
>
> `guidance_scale_tiles`: specific weights for classifier-free guidance in each tile. If `None`, the value provided in `guidance_scale` will be used.
>
> `seed_tiles`: specific seeds for the initialization latents in each tile. These will override the latents generated for the whole canvas using the standard `seed` parameter.
>
> `seed_tiles_mode`: either `"full"` `"exclusive"`. If `"full"`, all the latents affected by the tile be overriden. If `"exclusive"`, only the latents that are affected exclusively by this tile (and no other tiles) will be overrriden.
>
> `seed_reroll_regions`: a list of tuples in the form (start row, end row, start column, end column, seed) defining regions in pixel space for which the latents will be overriden using the given seed. Takes priority over `seed_tiles`.
>
> `cpu_vae`: the decoder from latent space to pixel space can require too mucho GPU RAM for large images. If you find out of memory errors at the end of the generation process, try setting this parameter to `True` to run the decoder in CPU. Slower, but should run without memory issues.

A script showing a more advanced use of this pipeline is available as [generate_grid.py](./generate_grid.py).

### StableDiffusionCanvasPipeline

The `StableDiffusionCanvasPipeline` works by defining a list of `Text2ImageRegion` objects that detail the region of influence of each diffuser. As an illustrative example, the heading image at this repo can be generated with the following code:

```python
from diffusers import LMSDiscreteScheduler
from mixdiff import StableDiffusionCanvasPipeline, Text2ImageRegion

# Creater scheduler and model (similar to StableDiffusionPipeline)
scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)
pipeline = StableDiffusionCanvasPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", scheduler=scheduler, use_auth_token=True).to("cuda:0")

# Mixture of Diffusers generation
image = pipeline(
    canvas_height=640,
    canvas_width=1408,
    regions=[
        Text2ImageRegion(0, 640, 0, 640, guidance_scale=8,
            prompt=f"A charming house in the countryside, by jakub rozalski, sunset lighting, elegant, highly detailed, smooth, sharp focus, artstation, stunning masterpiece"),
        Text2ImageRegion(0, 640, 384, 1024, guidance_scale=8,
            prompt=f"A dirt road in the countryside crossing pastures, by jakub rozalski, sunset lighting, elegant, highly detailed, smooth, sharp focus, artstation, stunning masterpiece"),
        Text2ImageRegion(0, 640, 768, 1408, guidance_scale=8,
            prompt=f"An old and rusty giant robot lying on a dirt road, by jakub rozalski, dark sunset lighting, elegant, highly detailed, smooth, sharp focus, artstation, stunning masterpiece"),
    ],
    num_inference_steps=50,
    seed=7178915308,
)["sample"][0]
```

`Image2Image` regions can also be added at any position, to use a particular image as guidance. In the following example we create a Christmas postcard by taking a photo of a building (available at this repo) and using it as a guidance in a region of the canvas.

```python
from PIL import Image
from diffusers import LMSDiscreteScheduler
from mixdiff import StableDiffusionCanvasPipeline, Text2ImageRegion, Image2ImageRegion, preprocess_image

# Creater scheduler and model (similar to StableDiffusionPipeline)
scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)
pipeline = StableDiffusionCanvasPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", scheduler=scheduler, use_auth_token=True).to("cuda:0")

# Load and preprocess guide image
iic_image = preprocess_image(Image.open("examples/IIC.png").convert("RGB"))

# Mixture of Diffusers generation
image = pipeline(
    canvas_height=800,
    canvas_width=352,
    regions=[
        Text2ImageRegion(0, 800, 0, 352, guidance_scale=8,
            prompt=f"Christmas postcard, a charming house in the countryside surrounded by snow, a giant christmas tree, under a starry night sky, by jakub rozalski and alayna danner and guweiz, elegant, highly detailed, smooth, sharp focus, artstation, stunning masterpiece"),
        Image2ImageRegion(800-352, 800, 0, 352, reference_image=iic_image, strength=0.8),
    ],
    num_inference_steps=57,
    seed=5525475061,
)["sample"][0]
```

![githubIIC](https://user-images.githubusercontent.com/9654655/218306373-fbae1381-178a-454c-89bf-0c299af4fb96.png)

The full list of arguments to a `StableDiffusionCanvasPipeline` is:

> `canvas_height`: height in pixels of the image to generate. Must be a multiple of 8.
>
> `canvas_width`: width in pixels of the image to generate. Must be a multiple of 8.
>
> `regions`: list of `Text2Image` or `Image2Image` diffusion regions (see below).
>
> `num_inference_steps`: number of diffusions steps.
>
> `seed`: general random seed to initialize latents.
>
> `reroll_regions`: list of `RerollRegion` regions in which to reroll latents (see below). Useful if you like the overall aspect of the generated image, but want to regenerate a specific region using a different random seed.
>
> `cpu_vae`: whether to perform encoder-decoder operations in CPU, even if the diffusion process runs in GPU. Use `cpu_vae=True` if you run out of GPU memory at the end of the generation process for large canvas dimensions, or if you create large `Image2Image` regions.
>
> `decode_steps`: if `True` the result will include not only the final image, but also all the intermediate steps in the generation. Note: this will greatly increase running times.

All regions are configured with the following parameters:

> `row_init`: starting row in pixel space (included). Must be a multiple of 8.
>
> `row_end`: end row in pixel space (not included). Must be a multiple of 8.
>
> `col_init`: starting column in pixel space (included). Must be a multiple of 8.
>
> `col_end`: end column in pixel space (not included). Must be a multiple of 8.
>
> `region_seed`: seed for random operations in this region
>
> `noise_eps`: deviation of a zero-mean gaussian noise to be applied over the latents in this region. Useful for slightly "rerolling" latents

Additionally, `Text2Image` regions use the following arguments:

> `prompt`: text prompt guiding the diffuser in this region
>
> `guidance_scale`: guidance scale of the diffuser in this region. If None, randomize.
>
> `mask_type`: kind of weight mask applied to this region, must be one of `["constant", gaussian", quartic"]`.
>
> `mask_weight`: global weights multiplier of the mask.

`Image2Image` regions are configured with the basic region parameters plus ther following:

> `reference_image`: image to use as guidance. Must be loaded as a PIL image and pre-processed using the `preprocess_image` function (see example above). It will be automatically rescaled to the shape of the region.
>
> `strength`: strength of the image guidance, must lie in the range `[0.0, 1.0]` (from no guidance to absolute priority of the original image).

Finally, `RerollRegions` accept the basic arguments plus the following:

> `reroll_mode`: kind of reroll to perform, either `reset` (completely reset latents with new ones) or `epsilon` (alter slightly the latents in the region).

## Citing and full technical details

If you find this repository useful, please be so kind to cite the corresponding paper, which also contains the full details about this method:

> Álvaro Barbero Jiménez. Mixture of Diffusers for scene composition and high resolution image generation. https://arxiv.org/abs/2302.02412

## Responsible use

The same recommendations as in Stable Diffusion apply, so please check the corresponding [model card](https://huggingface.co/CompVis/stable-diffusion-v1-4).

More broadly speaking, always bear this in mind: YOU are responsible for the content you create using this tool. Do not fully blame, credit, or place the responsibility on the software.

## Gallery

Here are some relevant illustrations I have created using this software (and putting quite a few hours into them!).

### Darkness Dawning

![Darkness Dawning](https://images-wixmp-ed30a86b8c4ca887773594c2.wixmp.com/f/cd1358aa-80d5-4c59-b95b-cdfde5dcc4f5/dfidq8n-6da9a886-9f1c-40ae-8341-d77af9552395.png?token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ1cm46YXBwOjdlMGQxODg5ODIyNjQzNzNhNWYwZDQxNWVhMGQyNmUwIiwiaXNzIjoidXJuOmFwcDo3ZTBkMTg4OTgyMjY0MzczYTVmMGQ0MTVlYTBkMjZlMCIsIm9iaiI6W1t7InBhdGgiOiJcL2ZcL2NkMTM1OGFhLTgwZDUtNGM1OS1iOTViLWNkZmRlNWRjYzRmNVwvZGZpZHE4bi02ZGE5YTg4Ni05ZjFjLTQwYWUtODM0MS1kNzdhZjk1NTIzOTUucG5nIn1dXSwiYXVkIjpbInVybjpzZXJ2aWNlOmZpbGUuZG93bmxvYWQiXX0.ff6XoVBPdUbcTLcuHUpQMPrD2TaXBM_s6HfRhsARDw0)

### Yog-Sothoth

![Yog-Sothoth](https://images-wixmp-ed30a86b8c4ca887773594c2.wixmp.com/f/cd1358aa-80d5-4c59-b95b-cdfde5dcc4f5/dfidsq4-174dd428-2c5a-48f6-a78f-9441fb3cffea.png?token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ1cm46YXBwOjdlMGQxODg5ODIyNjQzNzNhNWYwZDQxNWVhMGQyNmUwIiwiaXNzIjoidXJuOmFwcDo3ZTBkMTg4OTgyMjY0MzczYTVmMGQ0MTVlYTBkMjZlMCIsIm9iaiI6W1t7InBhdGgiOiJcL2ZcL2NkMTM1OGFhLTgwZDUtNGM1OS1iOTViLWNkZmRlNWRjYzRmNVwvZGZpZHNxNC0xNzRkZDQyOC0yYzVhLTQ4ZjYtYTc4Zi05NDQxZmIzY2ZmZWEucG5nIn1dXSwiYXVkIjpbInVybjpzZXJ2aWNlOmZpbGUuZG93bmxvYWQiXX0.X42zWgsk3lYnYwuEgkifRFRH2km-npHvrdleDN3m6bA)

### Looking through the eyes of giants

![Looking through the eyes of giants](https://user-images.githubusercontent.com/9654655/218307148-95ce88b6-b2a3-458d-b469-daf5bd56e3a7.jpg)

[Follow me on DeviantArt for more!](https://www.deviantart.com/albarji)

## Acknowledgements

First and foremost, my most sincere appreciation for the [Stable Diffusion team](https://stability.ai/blog/stable-diffusion-public-release) for releasing such an awesome model, and for letting me take part of the closed beta. Kudos also to the Hugging Face community and developers for implementing the [Diffusers library](https://github.com/huggingface/diffusers).

Thanks to Instituto de Ingeniería del Conocimiento and Grupo de Aprendizaje Automático (Universidad Autónoma de Madrid) for providing GPU resources for testing and experimenting this library.

Thanks also to the vibrant communities of the Stable Diffusion discord channel and [Lexica](https://lexica.art/), where I have learned about many amazing artists and styles. And to my friend Abril for sharing many tips on cool artists!
