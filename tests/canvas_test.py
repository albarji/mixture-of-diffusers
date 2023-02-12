from diffusers import LMSDiscreteScheduler
from PIL import Image
import pytest

from mixdiff import Image2ImageRegion, StableDiffusionCanvasPipeline, Text2ImageRegion, preprocess_image
from mixdiff.canvas import CanvasRegion

### CanvasRegion tests

@pytest.mark.parametrize("region_params", [
    {"row_init": 0, "row_end": 512, "col_init": 0, "col_end": 512}, 
    {"row_init": 0, "row_end": 256, "col_init": 0, "col_end": 512},
    {"row_init": 0, "row_end": 512, "col_init": 0, "col_end": 256},
    {"row_init": 0, "row_end": 512, "col_init": 0, "col_end": 512, "region_seed": 12345},
    {"row_init": 0, "row_end": 512, "col_init": 0, "col_end": 512, "noise_eps": 0.1} 
])
def test_create_canvas_region_correct(region_params):
    """Creating a correct canvas region with basic parameters works"""
    region = CanvasRegion(**region_params)
    assert region.row_init == region_params["row_init"]
    assert region.row_end == region_params["row_end"]
    assert region.col_init == region_params["col_init"]
    assert region.col_end == region_params["col_end"]
    assert region.height == region.row_end - region.row_init
    assert region.width == region.col_end - region.col_init

def test_create_canvas_region_eps():
    """Creating a correct canvas region works"""
    CanvasRegion(0, 512, 0, 512)
 
def test_create_canvas_region_non_multiple_size():
    """Creating a canvas region with sizes that are not a multiple of 8 fails"""
    with pytest.raises(ValueError):
        CanvasRegion(0, 17, 0, 15)

def test_create_canvas_region_negative_indices():
    """Creating a canvas region with negative indices fails"""
    with pytest.raises(ValueError):
        CanvasRegion(-512, 0, -256, 0)

def test_create_canvas_region_negative_eps():
    """Creating a canvas region with negative epsilon noise fails"""
    with pytest.raises(ValueError):
        CanvasRegion(0, 512, 0, 512, noise_eps=-3)

### Text2ImageRegion tests

@pytest.mark.parametrize("region_params", [
    {"row_init": 0, "row_end": 512, "col_init": 0, "col_end": 512},
    {"row_init": 0, "row_end": 512, "col_init": 0, "col_end": 512, "prompt": "Pikachu unit-testing Mixture of Diffusers"},
    {"row_init": 0, "row_end": 512, "col_init": 0, "col_end": 512, "prompt": "Pikachu unit-testing Mixture of Diffusers", "guidance_scale": 15.},
    {"row_init": 0, "row_end": 512, "col_init": 0, "col_end": 512, "prompt": "Pikachu unit-testing Mixture of Diffusers", "guidance_scale": 15., "mask_type": "constant", "mask_weight": 1.0},
    {"row_init": 0, "row_end": 512, "col_init": 0, "col_end": 512, "prompt": "Pikachu unit-testing Mixture of Diffusers", "guidance_scale": 15., "mask_type": "gaussian", "mask_weight": 0.75},
    {"row_init": 0, "row_end": 512, "col_init": 0, "col_end": 512, "prompt": "Pikachu unit-testing Mixture of Diffusers", "guidance_scale": 15., "mask_type": "quartic", "mask_weight": 1.5},
])
def test_create_text2image_region_correct(region_params):
    """Creating a Text2Image region with correct parameters works"""
    region = Text2ImageRegion(**region_params)
    if "prompt" in region_params: assert region.prompt == region_params["prompt"]
    if "guidance_scale" in region_params: assert region.guidance_scale == region_params["guidance_scale"]
    if "mask_type" in region_params: assert region.mask_type == region_params["mask_type"]
    if "mask_weight" in region_params: assert region.mask_weight == region_params["mask_weight"]

def test_create_text2image_region_negative_weight():
    """We can't specify a Text2Image region with mask weight"""
    with pytest.raises(ValueError):
        Text2ImageRegion(0, 512, 0, 512, prompt="Pikachu unit-testing Mixture of Diffusers", mask_type="gaussian", mask_weight=-0.1)

def test_create_text2image_region_unknown_mask():
    """We can't specify a Text2Image region with mask not in the recognized masks list"""
    with pytest.raises(ValueError):
        Text2ImageRegion(0, 512, 0, 512, prompt="Link unit-testing Mixture of Diffusers", mask_type="majora", mask_weight=1.0)

### Image2ImageRegion tests

@pytest.fixture(scope="session")
def base_image():
    return preprocess_image(Image.open("examples/IIC.png").convert("RGB"))

@pytest.mark.parametrize("region_params", [
    {"row_init": 0, "row_end": 512, "col_init": 0, "col_end": 512},
    {"row_init": 0, "row_end": 512, "col_init": 0, "col_end": 512, "strength": 0.1},
    {"row_init": 0, "row_end": 512, "col_init": 0, "col_end": 512, "strength": 0.9},
])
def test_create_image2image_region_correct(region_params, base_image):
    """Creating a Image2Image region with correct parameters works"""
    region = Image2ImageRegion(**region_params, reference_image=base_image)
    assert region.reference_image.shape == (1, 3, region.height, region.width)

@pytest.mark.parametrize("region_params", [
    {"row_init": 0, "row_end": 256, "col_init": 0, "col_end": 256, "strength": -0.3},
    {"row_init": 0, "row_end": 256, "col_init": 0, "col_end": 256, "strength": 1.1}
])
def test_create_image2image_region_negative_strength(region_params, base_image):
    """We can't specify an Image2Image region with strength values outside of [0, 1]"""
    with pytest.raises(ValueError):
        Image2ImageRegion(**region_params, reference_image=base_image)

### StableDiffusionCanvasPipeline tests

@pytest.fixture(scope="session")
def canvas_pipeline():
    return StableDiffusionCanvasPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4", 
        scheduler=LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000), 
        use_auth_token=True
    )

@pytest.fixture()
def basic_canvas_params():
    return {
        "canvas_height": 64,
        "canvas_width": 64,
        "regions": [
            Text2ImageRegion(0, 48, 0, 48, mask_type="gaussian", prompt="Something"),
            Text2ImageRegion(16, 64, 0, 48, mask_type="gaussian", prompt="Something else"),
            Text2ImageRegion(0, 48, 16, 64, mask_type="gaussian", prompt="Something more"),
            Text2ImageRegion(16, 64, 16, 64, mask_type="gaussian", prompt="One last thing"),
        ]
    }

@pytest.mark.parametrize("extra_canvas_params", [
    {"num_inference_steps": 1},
    {"num_inference_steps": 1, "cpu_vae": True},
])
def test_stable_diffusion_canvas_pipeline_correct(canvas_pipeline, basic_canvas_params, extra_canvas_params):
    """The StableDiffusionCanvasPipeline works for some correct configurations"""
    image = canvas_pipeline(**basic_canvas_params, **extra_canvas_params)["sample"][0]
    assert image.size == (64, 64)

@pytest.mark.parametrize("extra_canvas_params", [
    {"num_inference_steps": 3},
    {"num_inference_steps": 3, "cpu_vae": True},
])
def test_stable_diffusion_canvas_pipeline_image2image_correct(canvas_pipeline, basic_canvas_params, base_image, extra_canvas_params):
    """The StableDiffusionCanvasPipeline works for some correct configurations when including a Text2ImageRegion"""
    all_canvas_params = {**basic_canvas_params, **extra_canvas_params}
    all_canvas_params["regions"] += [Image2ImageRegion(16, 64, 0, 48, reference_image=base_image, strength=0.5)]

    image = canvas_pipeline(**all_canvas_params)["sample"][0]
    assert image.size == (64, 64)
