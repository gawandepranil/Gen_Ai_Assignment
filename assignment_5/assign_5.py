import torch
from diffusers import StableDiffusionPipeline

# model id
model_id = "runwayml/stable-diffusion-v1-5"

# load pipeline
pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16
)

# move model to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = pipe.to(device)

# prompt for artwork
prompt = "A futuristic city floating above the clouds, cyberpunk style, neon lights, ultra detailed digital art"

# generate image
image = pipe(prompt).images[0]

# save image
image.save("generated_art.png")

print("Image generated and saved!")