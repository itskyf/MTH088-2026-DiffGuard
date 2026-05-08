import torch
from diffusers import (
    ZImagePipeline,
    ZImageTransformer2DModel,
    GGUFQuantizationConfig,
)

gguf_path = "https://huggingface.co/unsloth/Z-Image-Turbo-GGUF/blob/main/z-image-turbo-Q4_K_S.gguf"

transformer = ZImageTransformer2DModel.from_single_file(
    gguf_path,
    quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
    torch_dtype=torch.bfloat16,
)

pipe = ZImagePipeline.from_pretrained(
    "Tongyi-MAI/Z-Image-Turbo",
    transformer=transformer,
    torch_dtype=torch.bfloat16,
    # low_cpu_mem_usage=False,
)

pipe.enable_model_cpu_offload()
# pipe.to("cuda")

prompt = "A cinematic photo of a cat holding a sign that says hello world"

image = pipe(
    prompt=prompt,
    height=1024,
    width=1024,
    num_inference_steps=9,  # official Z-Image Turbo: 9 => 8 DiT forwards
    guidance_scale=0.0,  # Turbo nên để 0
    generator=torch.Generator("cuda").manual_seed(42),
).images[0]

image.save("z-image-turbo-gguf-q4km.png")
