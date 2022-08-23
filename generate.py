import argparse
from diffusers import StableDiffusionPipeline, LMSDiscreteScheduler
import numpy as np
from PIL import Image
import torch

from diffusiontools.image2image import StableDiffusionImg2ImgPipeline
from diffusiontools.image2image import preprocess as preprocess_init

def generate_variations(prompt, height=512, width=768, seed=None, gc=None, steps=None, init_image=None):
    # Load model
    model_id = "CompVis/stable-diffusion-v1-4"
    scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)
    pipelinekind = StableDiffusionImg2ImgPipeline if init_image else StableDiffusionPipeline
    pipe = pipelinekind.from_pretrained(model_id, scheduler=scheduler, use_auth_token=True).to("cuda:0")

    # Load init image (if provided)
    if init_image:
        x0 = Image.open(init_image).convert("RGB")
        x0 = x0.resize((height, width))
        x0 = preprocess_init(x0)

    ## Generate variations
    while True:
        # Prepare parameters
        gc_image = gc if gc else np.random.randint(5, 30)
        steps_image = steps if steps else np.random.randint(50, 150)
        seed_image = seed if seed else np.random.randint(9999999999)
        pipeargs = {
            "guidance_scale": gc_image,
            "num_inference_steps": steps_image,
            "generator": torch.Generator("cuda").manual_seed(seed_image),
            "prompt": prompt,
        }
        if init_image:
            pipeargs = {**pipeargs, "init_image": x0}
        else:
            pipeargs = {**pipeargs, "height": height, "width": width}
        image = pipe(**pipeargs)["sample"][0]
        image.save(f"outputs/{height}x{width}_seed{seed_image}_gc{gc_image}_steps{steps_image}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate stable diffusion variations.')
    parser.add_argument('prompt', type=str, help='Prompt for generation')
    parser.add_argument('--height', type=int, default=512, help='Height in pixels of generated image')
    parser.add_argument('--width', type=int, default=512, help='Width in pixels of generated image')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for all generations (randomize seed if None)')
    parser.add_argument('--gc', type=float, default=None, help='Classifier-free Guidance scale (randomize if None)')
    parser.add_argument('--steps', type=int, default=None, help='Number of diffusion steps (randomize if None)')
    parser.add_argument('--init_image', type=str, default=None, help='Path to an image to use as initialization')
    args = parser.parse_args()
    generate_variations(
        prompt=args.prompt,
        height=args.height,
        width=args.width,
        seed=args.seed,
        gc=args.gc,
        steps=args.steps,
        init_image=args.init_image,
    )
