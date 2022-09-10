class StableDiffusionExtrasMixin:
    """Mixin providing additional convenience method to Stable Diffusion pipelines"""

    def decode_latents(self, latents, cpu_vae=False):
        """Decodes a given array of latents into pixel space"""
        # scale and decode the image latents with vae
        if cpu_vae:
            latents = latents.cpu()
            vae = self.vae.cpu()

        latents = 1 / 0.18215 * latents
        image = vae.decode(latents)

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()

        return self.numpy_to_pil(image)
