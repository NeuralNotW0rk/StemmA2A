import inspect
import tempfile
import queue
import threading
import uuid
import asyncio
from pathlib import Path
from dataclasses import replace

import torchaudio

from param_graph.elements.base_elements import GraphElement
from .engine import Engine
from .model_cache import ModelCache
from utils.uid import path_from_uid

class LocalEngine(Engine):
    def __init__(self, data_root: str = None):
        super().__init__()
        self.model_cache = ModelCache()
        
        if data_root:
            self.data_root = Path(data_root)
        else:
            # If no data root is provided, create a temporary one for this session.
            self.data_root = Path(tempfile.mkdtemp())
        
        self.data_root.mkdir(parents=True, exist_ok=True)
        print(f"LocalEngine initialized with data root: {self.data_root}")

        # --- Job Queue and Worker Setup ---
        self.job_queue = queue.Queue()
        self.job_statuses = {}
        self.worker_thread = threading.Thread(target=self._worker, daemon=True)
        self.worker_thread.start()
        
    async def register_model(self, adapter_name: str, **kwargs) -> GraphElement:
        """
        Registers a model by creating a temporary adapter and using it to 
        create the model element. The adapter is not cached.
        This remains a direct, synchronous-style async operation as it's not expected to be long-running.
        """
        adapter_class = self._get_adapter_class(adapter_name)
        adapter = adapter_class()
        return adapter.register_model(**kwargs)

    async def _generate_logic(self, **kwargs) -> GraphElement:
        """
        The actual generation logic. Gets a cached model adapter and uses it to generate an output.
        Saves the output to a persistent location within the data_root.
        """
        model_element = kwargs["model_element"]
        adapter_class = self._get_adapter_class(model_element.adapter)
        adapter = self.model_cache.get(model_element, adapter_class)

        artifact, tensor = adapter.generate(**kwargs)

        sample_rate = adapter.model_info.config["sample_rate"]
        
        local_path = self.data_root / path_from_uid(artifact.id)
        local_path.parent.mkdir(parents=True, exist_ok=True)

        # Create a named temp file with a .wav extension to satisfy torchaudio
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        
        try:
            # Save to the temp path (which torchaudio now recognizes as WAV)
            torchaudio.save(tmp_path, tensor, sample_rate)
            
            # Move the temp file to the final hash-based destination
            tmp_path.replace(local_path)
        except Exception as e:
            if tmp_path.exists():
                tmp_path.unlink()
            raise e

        # Update the artifact with the persistent path
        new_file_asset = replace(artifact.file, path=str(local_path))
        artifact = replace(artifact, file=new_file_asset)

        try:
            from .encoders.clap_encoder import CLAPEncoder
            encoder = CLAPEncoder()
            embedding = encoder.get_embedding(local_path)
            new_embeddings = artifact.embeddings.copy()
            new_embeddings[encoder.embedding_type] = embedding
            artifact = replace(artifact, embeddings=new_embeddings)
        except ImportError:
            print("CLAPEncoder not found, skipping embedding generation.")
        except Exception as e:
            print(f"Error during embedding generation: {e}")

        return artifact

    async def generate(self, **kwargs) -> GraphElement:
        """
        Public-facing generate method. For direct calls, it executes the logic.
        In the queued system, _generate_logic is called by the worker.
        """
        return await self._generate_logic(**kwargs)

    def _worker(self):
        """The worker function that processes jobs from the queue."""
        while True:
            job_id, operation_id, op_kwargs = self.job_queue.get()
            print(f"Worker: Picked up job {job_id} for operation '{operation_id}'")

            self.job_statuses[job_id] = {"status": "running"}

            try:
                # Get the actual operation function (e.g., self._generate_logic)
                if not hasattr(self, operation_id) or not operation_id.startswith('_'):
                    raise Exception(f"Operation '{operation_id}' is not a valid or public operation.")
                
                func = getattr(self, operation_id)
                
                # We need to run the async function in the current thread's event loop
                result = asyncio.run(func(**op_kwargs))

                self.job_statuses[job_id] = {"status": "completed", "result": result}
                print(f"Worker: Job {job_id} completed successfully.")

            except Exception as e:
                print(f"Worker: Job {job_id} failed. Error: {e}")
                import traceback
                traceback.print_exc()
                self.job_statuses[job_id] = {"status": "failed", "error": str(e)}
            finally:
                self.job_queue.task_done()

    async def execute(self, operation_id: str, **kwargs) -> str:
        """
        Queues an operation to be executed by the worker.
        Returns a job ID for status tracking.
        """
        job_id = str(uuid.uuid4())
        
        # For 'generate', we want the worker to call the core logic
        op_to_run = operation_id
        if operation_id == 'generate':
            op_to_run = '_generate_logic'

        self.job_queue.put((job_id, op_to_run, kwargs))
        self.job_statuses[job_id] = {"status": "pending"}
        print(f"Queued job {job_id} for operation '{operation_id}'")
        
        return job_id

    async def get_job_status(self, job_id: str) -> dict:
        """

        Returns the status and result (if available) of a job.
        If the job is complete, the result is serialized to a dictionary.
        """
        status = self.job_statuses.get(job_id)
        if not status:
            return {"status": "not_found"}
        
        # If the result is a GraphElement, convert it to a dict for the response
        if status.get("status") == "completed":
            result = status.get("result")
            if isinstance(result, GraphElement):
                status["result"] = result.to_dict()

        return status

    async def update_embedding(self, audio_artifact: GraphElement) -> GraphElement:
        """
        Calculates and adds embeddings to an existing audio artifact.
        """
        if not audio_artifact or not hasattr(audio_artifact, 'file'):
            raise ValueError("A valid audio artifact with a file asset is required.")

        audio_path = Path(audio_artifact.file.path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found at {audio_path}")

        try:
            from .encoders.clap_encoder import CLAPEncoder
            import torchaudio

            tensor, sample_rate = torchaudio.load(audio_path)
            
            encoder = CLAPEncoder()
            embedding = encoder.encode_audio(tensor, sample_rate)
            
            new_embeddings = audio_artifact.embeddings.copy()
            new_embeddings[encoder.embedding_type] = embedding
            
            updated_artifact = replace(audio_artifact, embeddings=new_embeddings)

            return updated_artifact

        except ImportError:
            print("CLAPEncoder not found, cannot generate embeddings.")
            return audio_artifact
        except Exception as e:
            print(f"Error during embedding update: {e}")
            return audio_artifact
