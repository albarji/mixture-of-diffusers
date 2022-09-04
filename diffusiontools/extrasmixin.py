class StableDiffusionExtrasMixin:
    """Mixin providing additional convenience method to Stable Diffusion pipelines"""

    def decode_latents(self, latents):
        """Decodes a given array of latents into pixel space"""
        # scale and decode the image latents with vae
        latents = 1 / 0.18215 * latents
        image = self.vae.decode(latents)

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()

        return self.numpy_to_pil(image)
