import inspect
import tempfile
from pathlib import Path
from dataclasses import replace

import torchaudio

from param_graph.elements.base_elements import GraphElement
from .engine import Engine
from .model_cache import ModelCache


class LocalEngine(Engine):
    def __init__(self):
        super().__init__()
        self.model_cache = ModelCache()
        
    async def register_model(self, adapter_name: str, **kwargs) -> GraphElement:
        """
        Registers a model by creating a temporary adapter and using it to 
        create the model element. The adapter is not cached.
        """
        adapter_class = self._get_adapter_class(adapter_name)
        adapter = adapter_class()
        return adapter.register_model(**kwargs)

    async def generate(self, **kwargs) -> GraphElement:
        """
        Gets a cached model adapter and uses it to generate an output.
        Saves the output to a temporary file and returns the artifact
        anchored to that file.
        """
        model_element = kwargs["model_element"]
        adapter_class = self._get_adapter_class(model_element.adapter)
        adapter = self.model_cache.get(model_element, adapter_class)

        artifact, tensor = adapter.generate(**kwargs)

        # We need to save the tensor to a temporary file.
        temp_dir = tempfile.TemporaryDirectory()
        temp_dir_path = Path(temp_dir.name)
        
        # We assume the model info needed for saving is on the adapter.
        # This might need to be more robust.
        sample_rate = adapter.model_info.config["sample_rate"]
        
        # Construct the path and save the file
        local_path = temp_dir_path / "output.wav"
        torchaudio.save(local_path, tensor, sample_rate)

        # Update the artifact with the temporary path
        new_file_asset = replace(artifact.file, path=str(local_path))
        artifact = replace(artifact, file=new_file_asset)
        
        # Store a reference to the TemporaryDirectory object to prevent
        # it from being garbage collected and deleting the file.
        artifact._temp_dir_ref = temp_dir

        return artifact

    async def execute(self, operation_id: str, **kwargs) -> GraphElement:
        """
        Dispatches the operation to a method on this class.
        """
        if not hasattr(self, operation_id):
            raise Exception(f"Operation '{operation_id}' not found on LocalEngine.")
        
        func = getattr(self, operation_id)

        # Basic validation: ensure it's a public method intended to be an operation.
        if operation_id.startswith('_') or not callable(func):
            raise Exception(f"Operation '{operation_id}' is not a valid operation.")

        # We assume all operations are async methods of this class.
        if inspect.iscoroutinefunction(func):
            return await func(**kwargs)
        else:
            # If we add non-async operations, we could handle them here.
            raise Exception(f"Operation '{operation_id}' is not an async function.")
