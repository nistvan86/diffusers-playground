from nicegui import ui, app, background_tasks, run
import services.model
import view.interface

# Start loading the model on startup
app.on_startup(lambda: background_tasks.create(run.io_bound(services.model.instance.load)))

if __name__ in {"__main__", "__mp_main__"}:
    # IMPORTANT: reload=False is required for our custom hot-reload to work 
    # without restarting the process (and thus reloading the model).
    ui.run(reload=False)
