from diffusers import StableDiffusionPipeline
import torch

pipe_oj = StableDiffusionPipeline.from_pretrained(
    "prompthero/openjourney-v4",
    torch_dtype=torch.float16,
    use_safetensors=True
).to("cuda")

pipe_oj.save_pretrained("models/openjourney")
