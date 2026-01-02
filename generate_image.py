from diffusers import StableDiffusionPipeline
import torch

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
)

pipe = pipe.to("cuda")
pipe.enable_attention_slicing()

prompt = "a shadow pic of girl in which she holding laddu gopal ji on her palm make a aesthetic pic" 


image = pipe(prompt).images[0]
image.save("generated_image.png")

print("Image generated successfully using CUDA!")
