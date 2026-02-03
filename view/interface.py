from nicegui import ui, background_tasks, run
import asyncio
import sys
import util
import services.model

# Access singleton
model = services.model.instance

@util.page('/')
def index():
    ui.dark_mode().enable()

    with ui.column().classes('w-full'):
        with ui.row().classes('w-full'):
            with ui.card().classes('w-full'):
                with ui.splitter(value=60).classes('w-full') as splitter:
                    with splitter.before as _:
                        # Prompt box
                        prompt = ui.textarea(label='Prompt', placeholder="Enter your prompt here.", value='Look up to the sky where a small one-man light plane drags a banner behind with the text "IT WORKS!", attached with strings.').classes('w-full q-pa-sm').props('outlined')
                    with splitter.after as _:
                        with ui.column().classes('q-pa-sm'):
                            # Seed
                            seed = ui.number(label='Seed', value=0, min=0, format='%d', precision=0).props('outlined')

                            with ui.row().classes('q-pb-sm'):
                                # Generate button
                                generate = ui.button("GO!").props("color=primary")
                                
                                if not model.is_loaded:
                                    generate.disable()

                                # Spinner while Z-Image loads
                                spinner = ui.spinner(size='2em')
                                if model.is_loaded:
                                    spinner.visible = False
        with ui.row():
            # Preview box
            preview = ui.image().props('bordered')
            preview.style('width: 1024px; height: 1024px').props('no-transition')


    def update_preview(base64_str: str) -> None:
        preview.set_source(base64_str)

    def on_generation_finished(base64_str: str) -> None:
        preview.set_source(base64_str)
        generate.enable()

    async def run_generation():
        generate.disable()
        # Run the IO/GPU bound task in a separate thread to not block the event loop
        await run.io_bound(
            model.generate, 
            prompt.value, 
            int(seed.value)
        )

    generate.on_click(run_generation)

    # Event subscriptions
    model.preview_event.subscribe(update_preview)
    model.finished_event.subscribe(on_generation_finished)

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
