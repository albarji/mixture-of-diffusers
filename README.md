# Mixture of diffusers for location-aware image generation

This repository holds various scripts and tools implementing a method for integrating a mixture of different diffusion processes collaborating to generate a single image. Each diffuser focuses on a particular region on the image, taking into account boundary effects to promote a smooth blending.

## Explanation

Current image generation methods, such as Stable Diffusion, struggle to position objects at specific locations. While the content of the generated image (somewhat) reflects the objects present in the prompt, it is difficult to frame the prompt in a way that creates an specific composition.

TODO: SD example

The method proposed here addresses this issue by using several diffusion processes in parallel, each configured with a specific prompt and settings, and focused on a particular region of the image.

TODO: SD-mixture example

## Usage details

This repository provides a new pipeline `StableDiffusionTilingPipeline` that extends the standard Stable Diffusion pipeline from [Diffusers](https://github.com/huggingface/diffusers). It features new options that allow defining the mixture of diffusers, which are distributed as a bidimensional grid over the image to be generated.


* **tile_height**: height in pixels of each grid tile.
* **tile_width**: width in pixels of each grid tile.
* **tile_row_overlap**: number of overlap pixels between tiles in consecutive rows.
* **tile_col_overlap**: number of overlap pixels between tiles in consecutive columns.
* **guidance_scale_tiles**: specific weights for classifier-free guidance in each tile.
* **guidance_scale_tiles**: specific weights for classifier-free guidance in each tile. If `None`, the value provided in `guidance_scale` will be used.
* **seed_tiles**: specific seeds for the initialization latents in each tile. These will override the latents generated for the whole canvas using the standard `seed` parameter.
* **seed_tiles_mode**: either `"full"` `"exclusive"`. If `"full"`, all the latents affected by the tile be overriden. If `"exclusive"`, only the latents that are affected exclusively by this tile (and no other tiles) will be overrriden.
* **seed_reroll_regions**: a list of tuples in the form (start row, end row, start column, end column, seed) defining regions in pixel space for which the latents will be overriden using the given seed. Has priority over `seed_tiles`.
