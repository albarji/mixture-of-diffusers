from dataclasses import dataclass
from enum import Enum
import inspect
import numpy as np
from numpy import pi, exp, sqrt
import torch
from tqdm.auto import tqdm
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer
from typing import List, Optional, Tuple, Union

from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.pipeline_utils import DiffusionPipeline
from diffusers.schedulers import DDIMScheduler, PNDMScheduler
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from diffusers.schedulers import LMSDiscreteScheduler

from diffusiontools.extrasmixin import StableDiffusionExtrasMixin


class MaskModes(Enum):
    """Modes in which the influence of diffuser is masked"""
    CONSTANT = "constant"
    GAUSSIAN = "gaussian"
    QUARTIC = "quartic"  # See https://en.wikipedia.org/wiki/Kernel_(statistics)


@dataclass
class CanvasRegion:
    """Class defining a rectangular region in the canvas"""
    row_init: int  # Region starting row in pixel space (included)
    row_end: int  # Region end row in pixel space (included)
    col_init: int  # Region starting column in pixel space (included)
    col_end: int  # Region end column in pixel space (included)

    def __post_init__(self):
        # Compute coordinates for this region in latent space
        self.latent_row_init = self.row_init // 8
        self.latent_row_end = self.row_end // 8
        self.latent_col_init = self.col_init // 8
        self.latent_col_end = self.col_end // 8


@dataclass
class DiffusionRegion(CanvasRegion):
    """Class for representing a rectangular region in the canvas where a particular diffuser will act"""
    prompt: str  # Text prompt guiding the diffuser in this region
    mask_type: MaskModes  # Kind of mask applied to this region
    mask_weight: float = 1.0  # Strength of the mask
    guidance_scale: float = 7.5  # Guidance scale of the diffuser in this region
    tokenized_prompt = None  # Tokenized prompt
    encoded_prompt = None  # Encoded prompt

    def tokenize_prompt(self, tokenizer):
        """Tokenizes the prompt for this diffusion region using a given tokenizer"""
        self.tokenized_prompt = tokenizer(self.prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")

    def encode_prompt(self, text_encoder, device):
        """Encodes the previously tokenized prompt for this diffusion region using a given encoder"""
        assert self.tokenized_prompt is not None, ValueError("Prompt in diffusion region must be tokenized before encoding")
        self.encoded_prompt = text_encoder(self.tokenized_prompt.input_ids.to(device))[0]


@dataclass
class MaskWeightsBuilder:
    """Auxiliary class to compute a tensor of weights for a given diffusion region"""
    latent_space_dim: int  # Size of the U-net latent space
    nbatch: int = 1  # Batch size in the U-net

    def compute_mask_weights(self, region: DiffusionRegion) -> torch.tensor:
        """Computes a tensor of weights for a given diffusion region"""
        MASK_BUILDERS = {
            MaskModes.CONSTANT.value: self._constant_weights,
            MaskModes.GAUSSIAN.value: self._gaussian_weights,
            MaskModes.QUARTIC.value: self._quartic_weights,
        }
        return MASK_BUILDERS[region.mask_type](region)

    def _constant_weights(self, region: DiffusionRegion) -> torch.tensor:
        """Computes a tensor of constant for a given diffusion region"""
        latent_width = region.latent_col_end - region.latent_col_init
        latent_height = region.latent_row_end - region.latent_row_init
        return torch.ones(self.nbatch, self.latent_space_dim, latent_height, latent_width) * region.mask_weight

    def _gaussian_weights(self, region: DiffusionRegion) -> torch.tensor:
        """Generates a gaussian mask of weights for tile contributions"""
        latent_width = region.latent_col_end - region.latent_col_init
        latent_height = region.latent_row_end - region.latent_row_init

        var = 0.01
        midpoint = (latent_width - 1) / 2  # -1 because index goes from 0 to latent_width - 1
        x_probs = [exp(-(x-midpoint)*(x-midpoint)/(latent_width*latent_width)/(2*var)) / sqrt(2*pi*var) for x in range(latent_width)]
        midpoint = (latent_height -1) / 2
        y_probs = [exp(-(y-midpoint)*(y-midpoint)/(latent_height*latent_height)/(2*var)) / sqrt(2*pi*var) for y in range(latent_height)]
        
        weights = np.outer(y_probs, x_probs) * region.mask_weight
        return torch.tile(torch.tensor(weights), (self.nbatch, self.latent_space_dim, 1, 1))

    def _quartic_weights(self, region: DiffusionRegion) -> torch.tensor:
        """Generates a quartic mask of weights for tile contributions
        
        The quartic kernel has bounded support over the diffusion region, and a smooth decay to the region limits.
        """
        quartic_constant = 15. / 16.        

        support = (np.array(range(region.latent_col_init, region.latent_col_end)) - region.latent_col_init) / (region.latent_col_end - region.latent_col_init - 1) * 1.99 - (1.99 / 2.)
        x_probs = quartic_constant * np.square(1 - np.square(support))
        support = (np.array(range(region.latent_row_init, region.latent_row_end)) - region.latent_row_init) / (region.latent_row_end - region.latent_row_init - 1) * 1.99 - (1.99 / 2.)
        y_probs = quartic_constant * np.square(1 - np.square(support))

        weights = np.outer(y_probs, x_probs) * region.mask_weight
        return torch.tile(torch.tensor(weights), (self.nbatch, self.latent_space_dim, 1, 1))
        

class StableDiffusionCanvasPipeline(DiffusionPipeline, StableDiffusionExtrasMixin):
    """Stable Diffusion pipeline that mixes several diffusers in the same canvas"""
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
        canvas_height: int,
        canvas_width: int,
        regions: List[DiffusionRegion],
        num_inference_steps: Optional[int] = 50,
        eta: Optional[float] = 0.0,
        seed: Optional[int] = None,
        seed_reroll_regions: Optional[List[Tuple[CanvasRegion, int]]] = None,
        cpu_vae: Optional[bool] = False,
    ):
        if seed_reroll_regions is None:
            seed_reroll_regions = []
        batch_size = 1

        # Create original noisy latents using the timesteps
        latents_shape = (batch_size, self.unet.in_channels, canvas_height // 8, canvas_width // 8)
        generator = torch.Generator("cuda").manual_seed(seed)
        latents = torch.randn(latents_shape, generator=generator, device=self.device)

        # Overwrite latents in seed reroll regions
        for region, seed_reroll in seed_reroll_regions:
            reroll_generator = torch.Generator("cuda").manual_seed(seed_reroll)
            region_shape = (latents_shape[0], latents_shape[1], region.latent_row_end - region.latent_row_init, region.latent_col_end - region.latent_col_init)
            latents[:, :, region.latent_row_init:region.latent_row_end, region.latent_col_init:region.latent_col_end] = torch.randn(region_shape, generator=reroll_generator, device=self.device)

        # Prepare scheduler
        accepts_offset = "offset" in set(inspect.signature(self.scheduler.set_timesteps).parameters.keys())
        extra_set_kwargs = {}
        if accepts_offset:
            extra_set_kwargs["offset"] = 1
        self.scheduler.set_timesteps(num_inference_steps, **extra_set_kwargs)
        # if we use LMSDiscreteScheduler, let's make sure latents are multiplied by sigmas
        if isinstance(self.scheduler, LMSDiscreteScheduler):
            latents = latents * self.scheduler.sigmas[0]

        # Prepare text embeddings
        for region in regions:
            region.tokenize_prompt(self.tokenizer)
            region.encode_prompt(self.text_encoder, self.device)

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = any([region.guidance_scale > 1.0 for region in regions])
        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance:
            for region in regions:
                max_length = region.tokenized_prompt.input_ids.shape[-1]
                uncond_input = self.tokenizer(
                    [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
                )
                uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]

                # For classifier free guidance, we need to do two forward passes.
                # Here we concatenate the unconditional and text embeddings into a single batch
                # to avoid doing two forward passes
                region.encoded_prompt = torch.cat([uncond_embeddings, region.encoded_prompt])

        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # Prepare mask of weights for each region
        mask_builder = MaskWeightsBuilder(latent_space_dim=self.unet.in_channels, nbatch=batch_size)
        mask_weights = [mask_builder.compute_mask_weights(region).to(self.device) for region in regions]

        # Diffusion timesteps
        for i, t in tqdm(enumerate(self.scheduler.timesteps)):
            # Diffuse each region            
            noise_preds_regions = []
            for region in regions:
                region_latents = latents[:, :, region.latent_row_init:region.latent_row_end, region.latent_col_init:region.latent_col_end]
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([region_latents] * 2) if do_classifier_free_guidance else region_latents
                if isinstance(self.scheduler, LMSDiscreteScheduler):
                    sigma = self.scheduler.sigmas[i]
                    # the model input needs to be scaled to match the continuous ODE formulation in K-LMS
                    latent_model_input = latent_model_input / ((sigma**2 + 1) ** 0.5)
                # predict the noise residual
                noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=region.encoded_prompt)["sample"]
                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    guidance = region.guidance_scale
                    noise_pred_region = noise_pred_uncond + guidance * (noise_pred_text - noise_pred_uncond)
                    noise_preds_regions.append(noise_pred_region)

            # Merge noise predictions for all tiles
            noise_pred = torch.zeros(latents.shape, device=self.device)
            contributors = torch.zeros(latents.shape, device=self.device)
            # Add each tile contribution to overall latents
            for region, noise_pred_region, mask_weights_region in zip(regions, noise_preds_regions, mask_weights):
                noise_pred[:, :, region.latent_row_init:region.latent_row_end, region.latent_col_init:region.latent_col_end] += noise_pred_region * mask_weights_region
                contributors[:, :, region.latent_row_init:region.latent_row_end, region.latent_col_init:region.latent_col_end] += mask_weights_region
            # Average overlapping areas with more than 1 contributor
            noise_pred /= contributors
            noise_pred = torch.nan_to_num(noise_pred)  # Replace NaNs by zeros: NaN can appear if a position is not covered by any DiffusionRegion

            # compute the previous noisy sample x_t -> x_t-1
            if isinstance(self.scheduler, LMSDiscreteScheduler):
                latents = self.scheduler.step(noise_pred, i, latents, **extra_step_kwargs)["prev_sample"]
            else:
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs)["prev_sample"]

        # scale and decode the image latents with vae
        image = self.decode_latents(latents, cpu_vae)

        return {"sample": image}
