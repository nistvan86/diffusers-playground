from nicegui import core, Event, ui, background_tasks, Client, app
import sys
import importlib
from watchfiles import awatch
from types import ModuleType

class ThreadSafeEvent(Event):
    def emit(self, *args, **kwargs):
        if core.loop:
            core.loop.call_soon_threadsafe(super().emit, *args, **kwargs)
        else:
            super().emit(*args, **kwargs)

_watched_modules = set()

def enable_hot_reload(module: ModuleType):
    module_name = module.__name__

    if module_name in _watched_modules:
        return
    _watched_modules.add(module_name)

    async def watch_reload():
        if not hasattr(module, '__file__'):
            print(f"Warning: Module {module_name} has no __file__ attribute, cannot watch.")
            return
            
        async for changes in awatch(module.__file__):
            try:
                print(f"Reloading {module_name} due to changes: {changes}")
                try:
                    importlib.reload(module)
                except Exception as e:
                     print(f"Error reloading module {module_name}: {e}")
                     continue

                for client in Client.instances.values():
                    with client:
                        ui.run_javascript('window.location.reload()')
            except Exception as e:
                print(f"Error watching/reloading {module_name}: {e}")

    app.on_startup(lambda: background_tasks.create(watch_reload()))

def hot_reload(module: ModuleType):
    def decorator(func):
        enable_hot_reload(module)
        return func
    return decorator

