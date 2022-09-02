import argparse
from diffusers import DDIMScheduler, LMSDiscreteScheduler
from diffusers import StableDiffusionPipeline
import numpy as np
from PIL import Image
import torch

from diffusiontools.image2image import StableDiffusionImg2ImgPipeline
from diffusiontools.image2image import preprocess as preprocess_init


def dummy_checker(images, **kwargs):
    "Dummy checker to override NSFW filter, which is sometimes annoying. Be responsible of your creations"
    return images, False


def generate_variations(prompt, height=512, width=512, seed=None, gc=None, steps=None, init_image=None, init_strength=None, n=25):
    # Load model
    model_id = "CompVis/stable-diffusion-v1-4"
    # Scheduler depends on pipeline. img2img requires a scheduler with integer timestep indexes,
    # not the case for LMSDiscreteScheduler
    scheduler = (
        DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False) if init_image
        else LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)
    )
    pipelinekind = StableDiffusionImg2ImgPipeline if init_image else StableDiffusionPipeline
    pipe = pipelinekind.from_pretrained(model_id, scheduler=scheduler, use_auth_token=True).to("cuda:0")
    # Override safety checker
    pipe.safety_checker = dummy_checker

    # Load init image (if provided)
    if init_image:
        x0 = Image.open(init_image).convert("RGB")
        x0 = x0.resize((width, height))
        x0 = preprocess_init(x0)

    ## Generate variations
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
        }
        outname = f"outputs/{prompt[0:100]}_{height}x{width}_seed{seed_image}_gc{gc_image}_steps{steps_image}"
        if init_image:
            strength_image = init_strength if init_strength is not None else np.random.randint(20, 80) / 100.
            pipeargs = {**pipeargs, "init_image": x0, "strength": 1. - strength_image}
            outname = outname + f"_initstr{strength_image}"
        else:
            pipeargs = {**pipeargs, "height": height, "width": width}
        image = pipe(**pipeargs)["sample"][0]
        image.save(f"{outname}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate stable diffusion variations.')
    parser.add_argument('prompt', type=str, help='Prompt for generation')
    parser.add_argument('--n', type=int, default=25, help='Number of variations to generate')
    parser.add_argument('--height', type=int, default=512, help='Height in pixels of generated image')
    parser.add_argument('--width', type=int, default=512, help='Width in pixels of generated image')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for all generations (randomize seed if None)')
    parser.add_argument('--gc', type=float, default=None, help='Classifier-free Guidance scale (randomize if None)')
    parser.add_argument('--steps', type=int, default=None, help='Number of diffusion steps (randomize if None)')
    parser.add_argument('--init_image', type=str, default=None, help='Path to an image to use as initialization')
    parser.add_argument('--init_strength', type=float, default=None, help='Strength of init image, between 0 and 1 (randomize if None)')
    args = parser.parse_args()
    generate_variations(
        prompt=args.prompt,
        height=args.height,
        width=args.width,
        seed=args.seed,
        gc=args.gc,
        steps=args.steps,
        init_image=args.init_image,
        init_strength=args.init_strength,
        n=args.n,
    )
