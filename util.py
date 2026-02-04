from nicegui import core, Event, ui, background_tasks, Client, app

import os
import importlib
from watchfiles import awatch
from types import ModuleType

class ThreadSafeEvent(Event):
    def emit(self, *args, **kwargs):
        if core.loop:
            core.loop.call_soon_threadsafe(super().emit, *args, **kwargs)
        else:
            super().emit(*args, **kwargs)

class HotReloader:
    def __init__(self):
        self._page_registry = {}  # file_path -> (module, [page_objects])

    def register(self, module: ModuleType, page_object):
        if not hasattr(module, '__file__'):
            print(f"Warning: Module {module.__name__} has no __file__ attribute, cannot watch.")
            return
        
        file_path = module.__file__
        if file_path not in self._page_registry:
            self._page_registry[file_path] = (module, [])
        
        self._page_registry[file_path][1].append(page_object)

    def start_watcher(self, directory: str):
        async def watch_reload():
            print(f"Watching directory {directory} for changes...")
            async for changes in awatch(directory):
                try:
                    for change_type, changed_file in changes:
                        if changed_file in self._page_registry:
                            print(f"Reloading {changed_file} due to changes")
                            module, page_objects = self._page_registry.pop(changed_file) # Unregister
                            
                            # Find clients to reload
                            clients_to_reload = []
                            for client in Client.instances.values():
                                if client.page in page_objects:
                                    clients_to_reload.append(client)
                            
                            try:
                                importlib.reload(module)
                            except Exception as e:
                                print(f"Error reloading module {module.__name__}: {e}")
                                # Put back the old registry, so it can be reloaded again
                                self._page_registry[changed_file] = (module, page_objects)
                                continue

                            # Notify clients
                            for client in clients_to_reload:
                                try:
                                    with client:
                                        ui.run_javascript('window.location.reload()')
                                except Exception as e:
                                    print(f"Error reloading client {client.id}: {e}")

                except Exception as e:
                    print(f"Error in hot reload watcher: {e}")

        background_tasks.create(watch_reload())


# Global singleton instance
hot_reloader = HotReloader()

def page(path: str, **kwargs):
    def decorator(func):
        import inspect
        module = inspect.getmodule(func)
        p = ui.page(path, **kwargs)
        hot_reloader.register(module, p)
        return p(func)
    return decorator
