import base64
import torch
from diffusers.pipelines.z_image.pipeline_z_image import ZImagePipeline
from diffusers.pipelines.z_image.pipeline_output import ZImagePipelineOutput
from typing import Dict, Any, Optional, cast, Callable
from io import BytesIO
from PIL import Image

class Latent2RGB:
    LATENT2RGB_COEFFICIENTS = [
        [-0.0056, 0.0405, 0.0824],
        [0.0470, 0.0437, 0.0898],
        [0.0459, -0.0523, -0.0236],
        [-0.0184, -0.0058, 0.0355],
        [0.0551, 0.0411, -0.0036],
        [-0.0300, 0.0084, -0.0115],
        [0.0280, 0.0895, 0.0636],
        [-0.0368, -0.0562, -0.0259],
        [-0.0439, 0.0039, 0.0924],
        [0.0987, 0.0592, -0.0801],
        [0.0165, 0.0545, 0.0573],
        [0.0933, 0.0345, 0.0342],
        [0.0446, 0.0430, 0.0413],
        [-0.0897, 0.0135, -0.0780],
        [0.0035, -0.0578, -0.0233],
        [-0.0826, -0.0540, -0.0286],
    ]
    LATENT2RGB_BIAS = [0.4848, 0.4871, 0.4499]

    def __init__(self) -> None:
        self.factors: Optional[torch.Tensor] = None
        self.bias: Optional[torch.Tensor] = None

    def __call__(self, latents: torch.Tensor) -> str:
        with torch.no_grad():
            if self.factors is None or self.bias is None:
                self.factors = torch.tensor(
                    self.LATENT2RGB_COEFFICIENTS, device=latents.device, dtype=latents.dtype
                )
                self.bias = torch.tensor(
                    self.LATENT2RGB_BIAS, device=latents.device, dtype=latents.dtype
                )

            # Latent2RGB Preview
            # [B, 16, H, W] -> [B, H, W, 16]
            latents_perm = latents.permute(0, 2, 3, 1)

            # [B, H, W, 16] @ [16, 3] -> [B, H, W, 3]
            image = torch.matmul(latents_perm, self.factors) + self.bias

            image = image.clamp(0, 1)
            image = image.cpu().float().numpy()
            image = (image * 255).round().astype("uint8")

            return self.imagetobase64(Image.fromarray(image[0]))

    @staticmethod
    def imagetobase64(image: Image.Image) -> str:
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        return f'data:image/png;base64,{base64.b64encode(buffered.getvalue()).decode("utf-8")}'

from util import ThreadSafeEvent

class ZImageModel:
    def __init__(self):
        self.pipe: Optional[ZImagePipeline] = None
        self.latent2rgb = Latent2RGB()
        self.should_stop_pipeline: bool = False
        self.is_loaded: bool = False
        self.loaded_event = ThreadSafeEvent()
        self.preview_event = ThreadSafeEvent()
        self.finished_event = ThreadSafeEvent()

    def load(self) -> None:
        if self.is_loaded:
            return
        # Use bfloat16 for optimal performance on supported GPU
        self.pipe = ZImagePipeline.from_pretrained(
            "Tongyi-MAI/Z-Image-Turbo",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=False,
        )
        self.pipe.to("cuda")
        print("ZImage loaded")
        self.is_loaded = True
        self.loaded_event.emit()

    def on_step_end(self, pipe: ZImagePipeline, step_index: int, timestep: int, callback_kwargs: Dict[str, Any]) -> Dict[str, Any]:
        if self.should_stop_pipeline:
            raise Exception("Pipeline stopped")

        preview = self.latent2rgb(callback_kwargs["latents"])
        self.preview_event.emit(preview)

        return callback_kwargs

    def generate(self, prompt: str, seed: int = 0) -> str:
        if self.pipe is None:
            raise RuntimeError("Pipeline not initialized")

        self.should_stop_pipeline = False

        # Generate Image
        output: ZImagePipelineOutput = cast(ZImagePipelineOutput, self.pipe(
            prompt=prompt,
            height=1024,
            width=1024,
            num_inference_steps=9,  # This actually results in 8 DiT forwards
            guidance_scale=0.0,  # Guidance should be 0 for the Turbo models
            generator=torch.Generator("cuda").manual_seed(seed),
            max_sequence_length=1024,
            callback_on_step_end=self.on_step_end,
        ))

        image = output.images[0]
        torch.cuda.empty_cache()
        
        result = Latent2RGB.imagetobase64(image)
        self.finished_event.emit(result)
        return result

# Singleton instance
instance = ZImageModel()
