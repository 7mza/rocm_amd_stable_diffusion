import random
import sys

import matplotlib.pyplot as plt
import torch
from diffusers import StableDiffusionPipeline

from authtoken import auth_token

INPUT = "gandalf"
RANDOM_INT = random.randint(0, sys.maxsize)
OUTPUT = "./output/%s_%s.png" % (INPUT.replace(" ", "_"), RANDOM_INT)

DEVICE = "cuda"
MODEL1 = "CompVis/stable-diffusion-v1-4"
MODEL2 = "runwayml/stable-diffusion-v1-5"
GUIDANCE_SCALE = 8.5
NUM_INFERENCE_STEPS = 75
HEIGHT = 512
WIDTH = 512  # 768
GENERATOR = torch.Generator(DEVICE).manual_seed(RANDOM_INT)

torch.cuda.empty_cache()  # GC

pipe = StableDiffusionPipeline.from_pretrained(
    MODEL2,
    revision="fp16",
    torch_dtype=torch.bfloat16,  # or float32, float16 = grey image
    use_auth_token=auth_token
).to(DEVICE)


def dummy(images, **kwargs):  # disable NSFW
    return images, False


pipe.safety_checker = dummy

image = pipe(
    INPUT,
    guidance_scale=GUIDANCE_SCALE,
    num_inference_steps=NUM_INFERENCE_STEPS,
    height=HEIGHT,
    width=WIDTH,
    generator=GENERATOR
).images[0]

image.save(OUTPUT)

im = plt.imread(OUTPUT)
_, ax = plt.subplots()
ax.imshow(im)
plt.show()

torch.cuda.empty_cache()  # GC
