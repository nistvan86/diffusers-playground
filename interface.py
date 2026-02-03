from nicegui import ui, background_tasks, run
import asyncio

def build_interface(model):
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
                    
                    if not model.is_loaded:
                        generate.disable()

                    # Spinner while Z-Image loads
                    spinner = ui.spinner(size='2em')
                    if model.is_loaded:
                        spinner.visible = False

    # Preview box
    preview = ui.interactive_image()

    def update_preview(base64_str: str) -> None:
        preview.set_source(base64_str)

    def on_generation_finished():
        generate.enable()

    async def run_generation():
        generate.disable()
        # Run the IO/GPU bound task in a separate thread to not block the event loop
        # We pass the update_preview callback to the model
        result_base64 = await run.io_bound(
            model.generate, 
            prompt.value, 
            int(seed.value), 
            on_preview=lambda p: update_preview(p) # This lambda will be called from the thread
            # NOTE: updating UI from a thread is generally unsafe in some GUIs, 
            # but NiceGUI/NiceGUI's binding might handle it or we might need `ui.run_javascript` or `ui.notify` wrappers.
            # Usually manipulating the element props is thread-safe enough or we might need `invoke_from_thread`.
            # Let's wrap it in a safer way if needed. 
            # `run.io_bound` runs in executor. 
        )
        update_preview(result_base64)
        on_generation_finished()

    generate.on_click(run_generation)

    # Event-driven model loading
    def enable_generation(_=None):
        if model.is_loaded:
            generate.enable()
            spinner.visible = False

    # Check immediately
    if model.is_loaded:
        enable_generation()
    else:
        # Simple subscription! Unsubscription is handled automatically by NiceGUI.
        model.loaded_event.subscribe(enable_generation)
