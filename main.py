from nicegui import ui, app, Client, background_tasks, run
import model
import interface
import importlib
from watchfiles import awatch

# Initialize the model singleton
z_image_model = model.ZImageModel()

@ui.page('/')
def index():
    interface.build_interface(z_image_model)

async def watch_reload():
    async for changes in awatch('interface.py'):
        print(f"Reloading interface due to changes: {changes}")
        importlib.reload(interface)
        for client in Client.instances.values():
            with client:
                ui.run_javascript('window.location.reload()')

def startup():
    # Start checking for reloads
    background_tasks.create(watch_reload())
    # Start loading the model in a separate thread
    background_tasks.create(run.io_bound(z_image_model.load))

app.on_startup(startup)

if __name__ in {"__main__", "__mp_main__"}:
    # IMPORTANT: reload=False is required for our custom hot-reload to work 
    # without restarting the process (and thus reloading the model).
    ui.run(reload=False)
