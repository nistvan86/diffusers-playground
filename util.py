from nicegui import core, Event, ui, background_tasks, Client, app, context
import sys
import os
import importlib
from watchfiles import awatch
from types import ModuleType
import functools

class ThreadSafeEvent(Event):
    def emit(self, *args, **kwargs):
        if core.loop:
            core.loop.call_soon_threadsafe(super().emit, *args, **kwargs)
        else:
            super().emit(*args, **kwargs)

class HotReloader:
    def __init__(self):
        self._watched_modules = set()

    def enable_hot_reload(self, module: ModuleType):
        module_name = module.__name__

        if module_name in self._watched_modules:
            return
        self._watched_modules.add(module_name)

        async def watch_reload():
            if not hasattr(module, '__file__'):
                print(f"Warning: Module {module_name} has no __file__ attribute, cannot watch.")
                return
            
            # Watch the directory containing the module
            directory = os.path.dirname(module.__file__)
            print(f"Watching directory {directory} for changes...")
                
            async for changes in awatch(directory):
                try:
                    print(f"Reloading {module_name} due to changes: {changes}")
                    try:
                        importlib.reload(module)
                    except Exception as e:
                         print(f"Error reloading module {module_name}: {e}")
                         continue

                    # Reload only clients associated with this module
                    for client in Client.instances.values():
                        try:
                            with client:
                                ui.run_javascript('window.location.reload()')
                        except Exception as e:
                            print(f"Error reloading client {client.id}: {e}")
                            
                except Exception as e:
                    print(f"Error watching/reloading {module_name}: {e}")

        app.on_startup(lambda: background_tasks.create(watch_reload()))


# Global singleton instance
hot_reloader = HotReloader()

def page(path: str, **kwargs):
    def decorator(func):
        import inspect
        module = inspect.getmodule(func)

        hot_reloader.enable_hot_reload(module)
        
        @ui.page(path, **kwargs)
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
             return func(*args, **kwargs)
             
        return wrapper
    return decorator
