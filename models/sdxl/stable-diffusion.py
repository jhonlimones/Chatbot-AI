from diffusers import StableDiffusionXLPipeline
import torch

pipe_sdxl = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True
).to("cuda")

pipe_sdxl.save_pretrained("models/sdxl")
