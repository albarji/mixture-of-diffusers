# Mixture of diffusers for location-aware image generation

![2022-10-12 15_35_27 305133_A charming house in the countryside, by jakub rozalski, sunset lighting, elegant, highly detailed, s_640x640_schelms_seed7178915308_gc8_steps50](https://user-images.githubusercontent.com/9654655/195362341-bc7766c2-f5c6-40f2-b457-59277aa11027.png)

This repository holds various scripts and tools implementing a method for integrating a mixture of different diffusion processes collaborating to generate a single image. Each diffuser focuses on a particular region on the image, taking into account boundary effects to promote a smooth blending.

## Explanation

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

The method proposed here strives to proved a better tool for image composition by using several diffusion processes in parallel, each configured with a specific prompt and settings, and focused on a particular region of the image. For example, the following are three outputs from this method, using the following prompts from left to right:

* "**A charming house in the countryside, by jakub rozalski, sunset lighting**, elegant, highly detailed, smooth, sharp focus, artstation, stunning masterpiece"
* "**A dirt road in the countryside crossing pastures, by jakub rozalski, sunset lighting**, elegant, highly detailed, smooth, sharp focus, artstation, stunning masterpiece"
* "**An old and rusty giant robot lying on a dirt road, by jakub rozalski, dark sunset lighting**, elegant, highly detailed, smooth, sharp focus, artstation, stunning masterpiece"

![2022-10-12 15_25_40 021063_A charming house in the countryside, by jakub rozalski, sunset lighting, elegant, highly detailed, s_640x640_schelms_seed9764851938_gc8_steps50](https://user-images.githubusercontent.com/9654655/195362152-6f3af44d-cf8a-494b-8cf8-36acd8f86871.png)
![2022-10-12 15_32_11 563087_A charming house in the countryside, by jakub rozalski, sunset lighting, elegant, highly detailed, s_640x640_schelms_seed2096547054_gc8_steps50](https://user-images.githubusercontent.com/9654655/195362315-8c2d01a8-62f2-4d96-90ca-9ad22f69398e.png)
![2022-10-12 15_35_27 305133_A charming house in the countryside, by jakub rozalski, sunset lighting, elegant, highly detailed, s_640x640_schelms_seed7178915308_gc8_steps50](https://user-images.githubusercontent.com/9654655/195362341-bc7766c2-f5c6-40f2-b457-59277aa11027.png)

The mixture of diffusion processes is done in a way that harmonizes the generation process, preventing "seam" effects in the generated image.

Using several diffusion processes in parallel has also practical advantages when generating very large images, as the GPU memory requirements are similar to that of generating an of the size of a single tile.

## Usage

This repository provides a new pipeline `StableDiffusionTilingPipeline` that extends the standard Stable Diffusion pipeline from [Diffusers](https://github.com/huggingface/diffusers). It features new options that allow defining the mixture of diffusers, which are distributed as a bidimensional grid over the image to be generated.

In the following example a grid of 10x1 tiles is configured to generate a forest in changing styles:

![gridExampleLabeled](https://user-images.githubusercontent.com/9654655/195371664-54d8a599-25d8-46ba-b823-3c7726ecb6ff.png)

The `StableDiffusionTilingPipeline` is configured to use 10 prompts (from left to right):

* "a forest, **ukiyo-e**, intricate, elegant, highly detailed, smooth, sharp focus, artstation, stunning masterpiece, impressive colors",
* "a forest, **ukiyo-e**, intricate, elegant, highly detailed, smooth, sharp focus, artstation, stunning masterpiece, impressive colors",
* "a forest, **by velazquez**, intricate, elegant, highly detailed, smooth, sharp focus, artstation, stunning masterpiece, impressive colors",
* "a forest, **by velazquez**, intricate, elegant, highly detailed, smooth, sharp focus, artstation, stunning masterpiece, impressive colors",
* "a forest, **impressionist style by van gogh**, intricate, elegant, highly detailed, smooth, sharp focus, artstation, stunning masterpiece, impressive colors",
* "a forest, **impressionist style by van gogh**, intricate, elegant, highly detailed, smooth, sharp focus, artstation, stunning masterpiece, impressive colors",
* "a forest, **cubist style by Pablo Picasso** intricate, elegant, highly detailed, smooth, sharp focus, artstation, stunning masterpiece, impressive colors",
* "a forest, **cubist style by Pablo Picasso** intricate, elegant, highly detailed, smooth, sharp focus, artstation, stunning masterpiece, impressive colors",
* "a forest, **80s synthwave style**, intricate, elegant, highly detailed, smooth, sharp focus, artstation, stunning masterpiece, impressive colors",
* "a forest, **80s synthwave style**, intricate, elegant, highly detailed, smooth, sharp focus, artstation, stunning masterpiece, impressive colors"

Each tile takes a shape of 768x512 pixels, and tiles overlap 256 pixels to avoid seam effects.

The corresponding settings file for generating this image would be:

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

and you can try it running `python generate_grid_from_json.py examples/linearForest.json`.

You can create your own settings files and use `generate_grid_from_json.py` to generate your image! Alternatively, you implement you own python script making use of the `StableDiffusionTilingPipeline`. You can find an example of such a script in `generate_grid.py`. The full list of arguments is:

* **prompt**: either a single string (no tiling) or a list of lists with all the prompts to use (one list for each row of tiles). This will also define the tiling structure.
* **num_inference_steps**: number of diffusions steps.
* **guidance_scale**: classifier-free guidance.
* **tile_height**: height in pixels of each grid tile.
* **tile_width**: width in pixels of each grid tile.
* **tile_row_overlap**: number of overlap pixels between tiles in consecutive rows.
* **tile_col_overlap**: number of overlap pixels between tiles in consecutive columns.
* **guidance_scale_tiles**: specific weights for classifier-free guidance in each tile.
* **guidance_scale_tiles**: specific weights for classifier-free guidance in each tile. If `None`, the value provided in `guidance_scale` will be used.
* **seed_tiles**: specific seeds for the initialization latents in each tile. These will override the latents generated for the whole canvas using the standard `seed` parameter.
* **seed_tiles_mode**: either `"full"` `"exclusive"`. If `"full"`, all the latents affected by the tile be overriden. If `"exclusive"`, only the latents that are affected exclusively by this tile (and no other tiles) will be overrriden.
* **seed_reroll_regions**: a list of tuples in the form (start row, end row, start column, end column, seed) defining regions in pixel space for which the latents will be overriden using the given seed. Has priority over `seed_tiles`.
* **cpu_vae**: the decoder from latent space to pixel space can require too mucho GPU RAM for large images. If you find out of memory errors at the end of the generation process, try setting this parameter to `True` to run the decoder in CPU. Slower, but should run without memory issues.
