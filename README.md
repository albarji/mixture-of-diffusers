# Mixture of diffusers for location-aware image generation

This repository holds various scripts and tools implementing a method for integrating a mixture of different diffusion processes collaborating to generate a single image. Each diffuser focuses on a particular region on the image, taking into account boundary effects to promote a smooth blending.

## Explanation

Current image generation methods, such as Stable Diffusion, struggle to position objects at specific locations. While the content of the generated image (somewhat) reflects the objects present in the prompt, it is difficult to frame the prompt in a way that creates an specific composition.

TODO: example

The method proposed here addresses this issue by using several diffusion processes in parallel, each configured with a specific prompt and settings, and focused on a particular region of the image.
