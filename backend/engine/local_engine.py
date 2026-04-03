import io
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

        # --- Encoder Setup ---
        self.encoder = None
        try:
            from .encoders.clap_encoder import CLAPEncoder
            self.encoder = CLAPEncoder()
            print("CLAPEncoder initialized.")
        except ImportError:
            print("CLAPEncoder not found, skipping embedding functionality.")
        except Exception as e:
            print(f"Error initializing CLAPEncoder: {e}")
        
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

        # Save to an in-memory buffer to avoid issues with filenames in soundfile
        buffer = io.BytesIO()
        torchaudio.save(buffer, tensor, sample_rate, format="wav")

        # Write the buffer's content to the final destination file
        local_path.write_bytes(buffer.getvalue())

        # Update the artifact with the persistent path
        new_file_asset = replace(artifact.file, path=str(local_path))
        artifact = replace(artifact, file=new_file_asset)

        if self.encoder:
            try:
                embedding = self.encoder.get_embedding(local_path)
                new_embeddings = artifact.embeddings.copy()
                new_embeddings[self.encoder.embedding_type] = embedding.tolist()
                artifact = replace(artifact, embeddings=new_embeddings)
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

            # Check if the job was cancelled while in the queue
            if self.job_statuses.get(job_id, {}).get("status") == "cancelled":
                print(f"Worker: Job {job_id} was cancelled before execution. Skipping.")
                self.job_queue.task_done()
                continue

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
        job_id = kwargs.pop('job_id', str(uuid.uuid4()))
        
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

    async def cancel_job(self, job_id: str):
        """
        Requests cancellation of a job.
        """
        status_info = self.job_statuses.get(job_id, {})
        current_status = status_info.get("status")

        if not current_status or current_status in ["completed", "failed", "cancelled"]:
            print(f"Job {job_id} is already in a terminal state ('{current_status}'). Cannot cancel.")
            return

        if current_status in ["pending", "running"]:
            self.job_statuses[job_id] = {"status": "cancelled"}
            print(f"Job {job_id} cancellation requested. Status set to 'cancelled'.")

    async def update_embedding(self, audio_artifact: GraphElement) -> GraphElement:
        """
        Calculates and adds embeddings to an existing audio artifact.
        """
        if not self.encoder:
            print("CLAPEncoder not available, cannot update embeddings.")
            return audio_artifact

        if not audio_artifact or not hasattr(audio_artifact, 'file'):
            raise ValueError("A valid audio artifact with a file asset is required.")

        audio_path = Path(audio_artifact.file.path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found at {audio_path}")

        try:
            import torchaudio

            tensor, sample_rate = torchaudio.load(audio_path)
            
            embedding = self.encoder.encode_audio(tensor, sample_rate)
            
            new_embeddings = audio_artifact.embeddings.copy()
            new_embeddings[self.encoder.embedding_type] = embedding
            
            updated_artifact = replace(audio_artifact, embeddings=new_embeddings)

            return updated_artifact

        except Exception as e:
            print(f"Error during embedding update: {e}")
            return audio_artifact
