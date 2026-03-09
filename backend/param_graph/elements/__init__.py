import pkgutil
import importlib

# Recursively walk all packages and modules from this directory
# and import them. This ensures all @register decorators are run.
# The prefix ensures that the imports are absolute from the project's
# perspective, which is more robust.
for loader, module_name, is_pkg in pkgutil.walk_packages(__path__, prefix=f'{__package__}.'):
    # We don't need to import __init__ files
    if '__init__' not in module_name:
        importlib.import_module(module_name)
