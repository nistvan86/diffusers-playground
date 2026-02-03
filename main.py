import base64

import torch
from diffusers.pipelines.z_image.pipeline_z_image import ZImagePipeline
from diffusers.pipelines.z_image.pipeline_output import ZImagePipelineOutput

from typing import Dict, Any, Optional, cast

from io import BytesIO
from nicegui import Event, ui, app, background_tasks, run
from PIL import Image

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

pipe: Optional[ZImagePipeline] = None

z_image_loaded_event = Event()
z_image_preview_update_event = Event[str]()
z_image_loaded: bool = False
z_image_generator_running: bool = False
z_image_generator_finished = Event()

def imagetobase64(image: Image.Image) -> str:
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return f'data:image/png;base64,{base64.b64encode(buffered.getvalue()).decode("utf-8")}'

class Latent2RGB:
    def __init__(self) -> None:
        self.factors: Optional[torch.Tensor] = None
        self.bias: Optional[torch.Tensor] = None

    def __call__(self, latents: torch.Tensor) -> str:
        with torch.no_grad():
            if self.factors is None or self.bias is None:
                self.factors = torch.tensor(
                    LATENT2RGB_COEFFICIENTS, device=latents.device, dtype=latents.dtype
                )
                self.bias = torch.tensor(
                    LATENT2RGB_BIAS, device=latents.device, dtype=latents.dtype
                )

            # Latent2RGB Preview
            # [B, 16, H, W] -> [B, H, W, 16]
            latents_perm = latents.permute(0, 2, 3, 1)

            # [B, H, W, 16] @ [16, 3] -> [B, H, W, 3]
            image = torch.matmul(latents_perm, self.factors) + self.bias

            image = image.clamp(0, 1)
            image = image.cpu().float().numpy()
            image = (image * 255).round().astype("uint8")

            return imagetobase64(Image.fromarray(image[0]))

latent2rgb = Latent2RGB()

should_stop_pipeline: bool = False

def on_step_end(pipe: Any, step_index: int, timestep: Any, callback_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    global should_stop_pipeline

    if should_stop_pipeline:
        pipe.interrupt = True
        return callback_kwargs

    preview = latent2rgb(callback_kwargs["latents"])
    z_image_preview_update_event.emit(preview)

    return callback_kwargs


def generate_zimage(prompt: str, seed: int = 0) -> None:
    global pipe
    global z_image_generator_running

    if pipe is None:
        print("Pipeline not initialized")
        return

    # Generate Image
    output: ZImagePipelineOutput = cast(ZImagePipelineOutput, pipe(
        prompt=prompt,
        height=1024,
        width=1024,
        num_inference_steps=9,  # This actually results in 8 DiT forwards
        guidance_scale=0.0,  # Guidance should be 0 for the Turbo models
        generator=torch.Generator("cuda").manual_seed(seed),
        max_sequence_length=1024,
        callback_on_step_end=on_step_end,
    ))

    image = output.images[0]

    torch.cuda.empty_cache()

    z_image_preview_update_event.emit(imagetobase64(image))

    z_image_generator_running = False
    z_image_generator_finished.emit(None)


def init_zimage() -> None:
    global pipe
    global z_image_loaded

    # Use bfloat16 for optimal performance on supported GPU
    pipe = ZImagePipeline.from_pretrained(
        "Tongyi-MAI/Z-Image-Turbo",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=False,
    )
    pipe.to("cuda")

    print("ZImage loaded")

    z_image_loaded = True
    z_image_loaded_event.emit(None)

@ui.page("/")
def page() -> None:
    global z_image_loaded
    global z_image_generator_running

    ui.dark_mode().enable()

    with ui.splitter(value=60).classes('w-full') as splitter:
        with splitter.before as _:
            # Prompt box
            prompt = ui.textarea(label='Prompt', placeholder='enter your prompt here').classes('w-full')
        with splitter.after as _:
            with ui.column():
                # Seed
                seed = ui.number(label='Seed', value=0, min=0, format='%d', precision=0)

                with ui.row():
                    # Generate button
                    generate = ui.button("Generate")
                    generate.props("flat")
                    generate.disable()

                    # Spinner while Z-Image loads
                    spinner = ui.spinner(size='2em')

    async def start_generate() -> None:
        generate.disable()
        background_tasks.create(run.io_bound(generate_zimage, prompt.value, int(seed.value)))

    generate.on_click(start_generate)
    z_image_generator_finished.subscribe(lambda: generate.enable())

    def on_z_image_loaded() -> None:
        generate.enable()
        spinner.visible = False

    if z_image_loaded:
        on_z_image_loaded()

    if z_image_loaded and not z_image_generator_running:
        generate.enable()

    z_image_loaded_event.subscribe(on_z_image_loaded)

    # Preview box
    preview = ui.interactive_image()
    def update_preview(base64: str) -> None:
        preview.set_source(base64)
    z_image_preview_update_event.subscribe(update_preview)  # type: ignore

def root():
    ui.sub_pages({
            "/": page,
    })

def startup():
    background_tasks.create(run.io_bound(init_zimage))
    pass

app.on_startup(startup)

ui.run(root, reload=True)
