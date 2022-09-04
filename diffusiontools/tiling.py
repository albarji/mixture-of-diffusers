import inspect
from typing import List, Optional, Union

import torch
import torch.nn.functional as F

import PIL
from tqdm.auto import tqdm
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer

from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.pipeline_utils import DiffusionPipeline
from diffusers.schedulers import DDIMScheduler, PNDMScheduler
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from diffusers.schedulers import LMSDiscreteScheduler

from diffusiontools.extrasmixin import StableDiffusionExtrasMixin


class StableDiffusionTilingPipeline(DiffusionPipeline, StableDiffusionExtrasMixin):
    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: Union[DDIMScheduler, PNDMScheduler],
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPFeatureExtractor,
    ):
        super().__init__()
        scheduler = scheduler.set_format("pt")
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
        )

    @torch.no_grad()
    def __call__(
        self,
        prompt: List[List[str]],
        num_inference_steps: Optional[int] = 50,
        guidance_scale: Optional[float] = 7.5,
        eta: Optional[float] = 0.0,
        generator: Optional[torch.Generator] = None,
        output_type: Optional[str] = "pil",
        tile_height: Optional[int] = 512,
        tile_width: Optional[int] = 512,
        tile_overlap: Optional[int] = 256
    ):

        if not isinstance(prompt, list) or not all(isinstance(row, list) for row in prompt):
            raise ValueError(f"`prompt` has to be a list of lists but is {type(prompt)}")
        grid_rows = len(prompt)
        grid_cols = len(prompt[0])
        if not all(len(row) == grid_cols for row in prompt):
            raise ValueError(f"All prompt rows must have the same number of prompt columns")
        batch_size = 1

        # set timesteps
        accepts_offset = "offset" in set(inspect.signature(self.scheduler.set_timesteps).parameters.keys())
        extra_set_kwargs = {}
        offset = 0
        if accepts_offset:
            offset = 1
            extra_set_kwargs["offset"] = 1

        # create original noisy latents using the timesteps
        height = tile_height + (grid_rows - 1) * (tile_height - tile_overlap)
        width = tile_width + (grid_cols - 1) * (tile_width - tile_overlap)
        latents_shape = (batch_size, self.unet.in_channels, height // 8, width // 8)
        latents = torch.randn(latents_shape, generator=generator, device=self.device)

        # set timesteps
        accepts_offset = "offset" in set(inspect.signature(self.scheduler.set_timesteps).parameters.keys())
        extra_set_kwargs = {}
        if accepts_offset:
            extra_set_kwargs["offset"] = 1

        self.scheduler.set_timesteps(num_inference_steps, **extra_set_kwargs)

        # if we use LMSDiscreteScheduler, let's make sure latents are mulitplied by sigmas
        if isinstance(self.scheduler, LMSDiscreteScheduler):
            latents = latents * self.scheduler.sigmas[0]

        # get prompts text embeddings
        text_input = [
            [
                self.tokenizer(
                    col,
                    padding="max_length",
                    max_length=self.tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )
                for col in row
            ]
            for row in prompt
        ]
        text_embeddings = [
            [
                self.text_encoder(col.input_ids.to(self.device))[0]
                for col in row
            ]
            for row in text_input
        ]

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0
        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance:
            for i in range(grid_rows):
                for j in range(grid_cols):
                    max_length = text_input[i][j].input_ids.shape[-1]
                    uncond_input = self.tokenizer(
                        [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
                    )
                    uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]

                    # For classifier free guidance, we need to do two forward passes.
                    # Here we concatenate the unconditional and text embeddings into a single batch
                    # to avoid doing two forward passes
                    text_embeddings[i][j] = torch.cat([uncond_embeddings, text_embeddings[i][j]])

        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # Mask for tile weights strenght
        tile_weights = self._gaussian_weights(tile_width, tile_height, batch_size)

        # Diffusion timesteps
        for i, t in tqdm(enumerate(self.scheduler.timesteps)):
            # Diffuse each tile
            noise_preds = []
            for row in range(grid_rows):
                noise_preds_row = []
                for col in range(grid_cols):
                    px_row_init, px_row_end, px_col_init, px_col_end = _tile2latent_indices(row, col, tile_width, tile_height, tile_overlap)
                    tile_latents = latents[:, :, px_row_init:px_row_end, px_col_init:px_col_end]
                    # expand the latents if we are doing classifier free guidance
                    latent_model_input = torch.cat([tile_latents] * 2) if do_classifier_free_guidance else tile_latents
                    if isinstance(self.scheduler, LMSDiscreteScheduler):
                        sigma = self.scheduler.sigmas[i]
                        # the model input needs to be scaled to match the continuous ODE formulation in K-LMS
                        latent_model_input = latent_model_input / ((sigma**2 + 1) ** 0.5)
                    # predict the noise residual
                    noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings[row][col])["sample"]
                    # perform guidance
                    if do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred_tile = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                        noise_preds_row.append(noise_pred_tile)
                noise_preds.append(noise_preds_row)
            # Stitch noise predictions for all tiles
            noise_pred = torch.zeros(latents.shape, device=self.device)
            contributors = torch.zeros(latents.shape, device=self.device)
            # Add each tile contribution to overall latents
            for row in range(grid_rows):
                for col in range(grid_cols):
                    px_row_init, px_row_end, px_col_init, px_col_end = _tile2latent_indices(row, col, tile_width, tile_height, tile_overlap)
                    noise_pred[:, :, px_row_init:px_row_end, px_col_init:px_col_end] += noise_preds[row][col] * tile_weights
                    contributors[:, :, px_row_init:px_row_end, px_col_init:px_col_end] += tile_weights
            # Average overlapping areas with more than 1 contributor
            noise_pred /= contributors

            # # Upper-left tile: copy noise as it is
            # noise_pred[0:tile_height, 0:tile_width] = noise_preds[0][0]
            # # First row after upper-left tile: average noise in overlap area, copy in rest
            # for col in range(1, grid_cols):
            #     noise_pred[0:tile_height, (tile_width*col-tile_overlap):(tile_width*col)] += noise_preds[0][col][:, 0:tile_overlap]
            #     noise_pred[0:tile_height, (tile_width*col-tile_overlap):(tile_width*col)] /= 2
            #     noise_pred[0:tile_height, tile_width-tile_overlap:tile_width]
            # for row in range(grid_rows):
            #     # First col of row: copy noise
            #     noise_pred[row * ]

            #     for col in range(grid_cols):

            # compute the previous noisy sample x_t -> x_t-1
            if isinstance(self.scheduler, LMSDiscreteScheduler):
                latents = self.scheduler.step(noise_pred, i, latents, **extra_step_kwargs)["prev_sample"]
            else:
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs)["prev_sample"]

        # scale and decode the image latents with vae # TODO: replace by utility function
        latents = 1 / 0.18215 * latents
        image = self.vae.decode(latents)

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()

        # run safety checker # TODO remove
        safety_cheker_input = self.feature_extractor(self.numpy_to_pil(image), return_tensors="pt").to(self.device)
        image, has_nsfw_concept = self.safety_checker(images=image, clip_input=safety_cheker_input.pixel_values)

        if output_type == "pil":
            image = self.numpy_to_pil(image)

        return {"sample": image, "nsfw_content_detected": has_nsfw_concept}

    def _gaussian_weights(self, tile_width, tile_height, nbatches):
        """Generates a gaussian mask of weights for tile contributions"""
        from numpy import pi, exp, sqrt
        import numpy as np

        latent_width = tile_width // 8
        latent_height = tile_height // 8

        var = 0.01
        midpoint = (latent_width - 1) / 2  # -1 because index goes from 0 to latent_width - 1
        x_probs = [exp(-(x-midpoint)*(x-midpoint)/(latent_width*latent_width)/(2*var)) / sqrt(2*pi*var) for x in range(latent_width)]
        midpoint = latent_height / 2
        y_probs = [exp(-(y-midpoint)*(y-midpoint)/(latent_height*latent_height)/(2*var)) / sqrt(2*pi*var) for y in range(latent_height)]
        
        weights = np.outer(y_probs, x_probs)
        return torch.tile(torch.tensor(weights, device=self.device), (nbatches, self.unet.in_channels, 1, 1))



def _tile2pixel_indices(tile_row, tile_col, tile_width, tile_height, tile_overlap):
    """Given a tile row and column numbers returns the range of pixels affected by that tiles in the overall image
    
    Returns a tuple with:
        - Starting coordinates of rows in pixel space
        - Ending coordinates of rows in pixel space
        - Starting coordinates of columns in pixel space
        - Ending coordinates of columns in pixel space
    """
    px_row_init = 0 if tile_row == 0 else tile_row * (tile_height - tile_overlap)
    px_row_end = px_row_init + tile_height
    px_col_init = 0 if tile_col == 0 else tile_col * (tile_width - tile_overlap)
    px_col_end = px_col_init + tile_width
    return px_row_init, px_row_end, px_col_init, px_col_end


def _tile2latent_indices(tile_row, tile_col, tile_width, tile_height, tile_overlap):
    """Given a tile row and column numbers returns the range of latents affected by that tiles in the overall image
    
    Returns a tuple with:
        - Starting coordinates of rows in latent space
        - Ending coordinates of rows in latent space
        - Starting coordinates of columns in latent space
        - Ending coordinates of columns in latent space
    """
    px_row_init, px_row_end, px_col_init, px_col_end = _tile2pixel_indices(tile_row, tile_col, tile_width, tile_height, tile_overlap)
    return px_row_init // 8, px_row_end // 8, px_col_init // 8, px_col_end // 8
