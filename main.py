from nicegui import ui, app, Client, background_tasks, run
import model
import interface
import util

# Initialize the model singleton
z_image_model = model.ZImageModel()

@ui.page('/')
@util.hot_reload(interface)
def index():
    interface.build_interface(z_image_model)

def startup():
    # Start loading the model in a separate thread
    background_tasks.create(run.io_bound(z_image_model.load))

app.on_startup(startup)

if __name__ in {"__main__", "__mp_main__"}:
    # IMPORTANT: reload=False is required for our custom hot-reload to work 
    # without restarting the process (and thus reloading the model).
    ui.run(reload=False)
