from nicegui import core, Event, ui, background_tasks, Client, app, context
import sys
import importlib
from watchfiles import awatch
from types import ModuleType
from collections import defaultdict
from weakref import WeakSet
import functools

class ThreadSafeEvent(Event):
    def emit(self, *args, **kwargs):
        if core.loop:
            core.loop.call_soon_threadsafe(super().emit, *args, **kwargs)
        else:
            super().emit(*args, **kwargs)

_watched_modules = set()
_module_clients = defaultdict(WeakSet)

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

                # Reload only clients associated with this module
                for client in _module_clients[module_name]:
                    try:
                        with client:
                            ui.run_javascript('window.location.reload()')
                    except Exception as e:
                        print(f"Error reloading client {client.id}: {e}")
                        
            except Exception as e:
                print(f"Error watching/reloading {module_name}: {e}")

    app.on_startup(lambda: background_tasks.create(watch_reload()))

def hot_reload_page(path: str, module: ModuleType, **kwargs):
    def decorator(func):
        enable_hot_reload(module)
        
        @ui.page(path, **kwargs)
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
             client = context.client
             _module_clients[module.__name__].add(client)
             return func(*args, **kwargs)
             
        return wrapper
    return decorator
