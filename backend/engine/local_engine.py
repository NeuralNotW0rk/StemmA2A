import io
import os
import tempfile
import queue
import threading
import uuid
import asyncio
import gc
import torch
from pathlib import Path
from dataclasses import replace

import torchaudio

from param_graph.elements.base_elements import GraphElement
from .engine import Engine
from .model_cache import ModelCache
from utils.uid import path_from_uid
from utils.audio import save_audio

from diffracture import Actant
from diffracture.topology.grating import Grating as DiffractureGrating
from diffracture.analysis.clustering import FeatureClusteringPipeline

class LocalEngine(Engine):
    def __init__(self, data_root: str = None):
        super().__init__(data_root=data_root)
        
        self.data_root.mkdir(parents=True, exist_ok=True)
        print(f"LocalEngine initialized with data root: {self.data_root}")

        # --- Job Queue and Worker Setup ---
        self.job_queue = queue.Queue()
        self.job_statuses = {}
        self.is_sleeping = False
        self.idle_timeout = int(os.environ.get("ENGINE_IDLE_TIMEOUT", 6000))  # 30 minutes default
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

        # --- Shared Models Setup ---
        self.shared_models = []
        self._load_shared_models()

    def _load_shared_models(self):
        import json
        import shutil
        import traceback

        config_path_str = os.environ.get("SHARED_MODELS_CONFIG", "shared_models.json")
        config_path = Path(config_path_str)
        if not config_path.is_absolute():
            # Resolve relative to the backend directory
            config_path = Path(__file__).parent.parent / config_path_str

        if not config_path.exists():
            print(f"Shared models config not found at {config_path}. No shared models loaded.")
            return

        try:
            with open(config_path, "r") as f:
                config_data = json.load(f)
        except Exception as e:
            print(f"Failed to read shared models config: {e}")
            return

        for entry in config_data:
            name = entry.get("name")
            adapter_name = entry.get("adapter")
            model_type = entry.get("model_type")
            config_file_path = entry.get("config_path")
            checkpoint_file_path = entry.get("checkpoint_path")
            encoder_file_path = entry.get("encoder_path")

            if not name or not adapter_name or not config_file_path or not checkpoint_file_path:
                print(f"Skipping invalid shared model entry: {entry}")
                continue

            if not os.path.exists(config_file_path):
                print(f"Skipping shared model '{name}' because config path does not exist: {config_file_path}")
                continue
            if not os.path.exists(checkpoint_file_path):
                print(f"Skipping shared model '{name}' because checkpoint path does not exist: {checkpoint_file_path}")
                continue

            try:
                adapter_class = self._get_adapter_class(adapter_name)
                adapter = adapter_class()
                model_element = adapter.register_model(
                    name=name,
                    config_path=config_file_path,
                    checkpoint_path=checkpoint_file_path,
                    encoder_path=encoder_file_path,
                    model_type=model_type,
                )

                # Cache/symlink assets
                for asset_name, asset in model_element._iter_assets():
                    if asset and asset.path and asset.uid:
                        dest_path = Path(self.data_root) / path_from_uid(asset.uid)
                        dest_path.parent.mkdir(parents=True, exist_ok=True)
                        if not dest_path.exists():
                            try:
                                os.symlink(asset.path, dest_path)
                                print(f"Created symlink for shared model asset {asset.uid}: {asset.path} -> {dest_path}")
                            except Exception as sym_err:
                                try:
                                    shutil.copy(asset.path, dest_path)
                                    print(f"Copied shared model asset {asset.uid} to cache: {dest_path}")
                                except Exception as copy_err:
                                    print(f"Failed to copy shared model asset {asset.uid}: {copy_err}")

                self.shared_models.append(model_element)
                print(f"Successfully loaded shared model: {name} ({model_element.id})")
            except Exception as e:
                print(f"Failed to load shared model '{name}': {e}")
                traceback.print_exc()

    async def get_shared_models(self) -> list[dict]:
        return [model.to_dict() for model in self.shared_models]

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

        # Engine-level Grating intervention orchestration
        grating_elements = kwargs.get("grating_elements", [])
        grating_strengths = kwargs.get("grating_strengths", [])
        if grating_elements:
            if not hasattr(adapter, 'model') or adapter.model is None:
                raise RuntimeError(f"Adapter '{model_element.adapter}' does not expose a loaded 'model' for grating injection.")
            
            if not hasattr(adapter, 'actant'):
                adapter.actant = Actant(adapter.model)
                
            model_device = next(adapter.model.parameters()).device
            
            for i, grating_element in enumerate(grating_elements):
                strength = grating_strengths[i] if grating_strengths and i < len(grating_strengths) else 1.0
                print(f"Engine: Applying Grating '{grating_element.id}' with strength {strength}...")
                grating = DiffractureGrating.load(grating_element.file.path)
                
                # Apply dynamic overrides if present in the payload
                gratings_input = kwargs.get("gratings") or []
                for g_in in gratings_input:
                    if g_in.get("id") == grating_element.id:
                        overrides = g_in.get("overrides") or []
                        for o in overrides:
                            addr = o.get("address")
                            meta_overrides = o.get("metadata") or {}
                            if addr in grating.nodes:
                                print(f"Engine: Overriding grating element '{addr}' metadata with: {meta_overrides}")
                                grating.nodes[addr].metadata.update(meta_overrides)
                
                grating.to(model_device)
                adapter.actant.activate(grating, injection_strategy="hook", strength=strength)

        artifact, tensor = adapter.generate(**kwargs)

        if grating_elements:
            # Revert the model to its original state so it can remain safely in the cache
            adapter.actant.deactivate()

        local_path = self.data_root / path_from_uid(artifact.id)
        local_path.parent.mkdir(parents=True, exist_ok=True)

        if artifact.type == "image":
            from torchvision.utils import save_image
            save_image(tensor, local_path, format="png", normalize=True, value_range=(-1, 1))
        else:
            sample_rate = adapter.model_info.config["sample_rate"]
            save_audio(tensor, local_path, sample_rate, format="wav")

        # Update the artifact with the persistent path
        new_file_asset = replace(artifact.file, path=str(local_path))
        artifact = replace(artifact, file=new_file_asset)

        if self.encoder and artifact.type == "audio":
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

    async def _invert_logic(self, **kwargs) -> GraphElement:
        """
        The actual inversion logic. Gets a cached model adapter and uses it to perform DDIM inversion.
        Saves the resulting latent tensor to a persistent location.
        """
        model_element = kwargs["model_element"]
        adapter_class = self._get_adapter_class(model_element.adapter)
        adapter = self.model_cache.get(model_element, adapter_class)

        artifact, tensor = adapter.invert(**kwargs)

        local_path = self.data_root / path_from_uid(artifact.id)
        local_path.parent.mkdir(parents=True, exist_ok=True)

        # Save the latent tensor file
        torch.save(tensor.cpu(), local_path)

        # Update the artifact with the persistent path
        new_file_asset = replace(artifact.file, path=str(local_path))
        artifact = replace(artifact, file=new_file_asset)

        return artifact

    async def invert(self, **kwargs) -> GraphElement:
        return await self._invert_logic(**kwargs)

    def _worker(self):
        """The worker function that processes jobs from the queue."""
        while True:
            try:
                job_id, operation_id, op_kwargs = self.job_queue.get(timeout=self.idle_timeout)
                self.is_sleeping = False
            except queue.Empty:
                if not self.is_sleeping and len(self.model_cache.cache) > 0:
                    print(f"Worker: Idle for {self.idle_timeout} seconds. Entering sleep mode (clearing VRAM).")
                    self.model_cache.clear()
                    
                    # Force Python to collect garbage immediately
                    gc.collect()
                    # Force PyTorch to release cached VRAM back to the OS
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        
                    self.is_sleeping = True
                continue

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
        
        # Dynamically map the requested operation to its protected logic method
        op_to_run = f"_{operation_id}_logic"
        if not hasattr(self, op_to_run):
            raise ValueError(f"Operation '{operation_id}' is not supported by the LocalEngine.")

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
            embedding = self.encoder.get_embedding(str(audio_path))
            
            new_embeddings = audio_artifact.embeddings.copy()
            new_embeddings[self.encoder.embedding_type] = embedding.tolist()
            
            updated_artifact = replace(audio_artifact, embeddings=new_embeddings)

            return updated_artifact

        except Exception as e:
            print(f"Error during embedding update: {e}")
            return audio_artifact

    async def get_model_layers(self, model_element: GraphElement) -> list[dict]:
        adapter_class = self._get_adapter_class(model_element.adapter)
        adapter = self.model_cache.get(model_element, adapter_class)
        if not hasattr(adapter, 'model') or adapter.model is None:
            raise RuntimeError("Model failed to load or does not expose PyTorch module.")
            
        return self._extract_model_layers(adapter.model)

    async def cluster_features(self, model_element: GraphElement, address: str, num_clusters: int) -> list[int]:
        adapter_class = self._get_adapter_class(model_element.adapter)
        adapter = self.model_cache.get(model_element, adapter_class)
        

        print(f"Running dynamic FeatureClusteringPipeline on layer '{address}' with {num_clusters} clusters...")
        pipeline = FeatureClusteringPipeline(adapter.model, strategy_name="cnn")
        pipeline.collector.start_collecting([address])
        
        # Run dummy forward passes on StyleGAN/StableAudio to collect activations
        with torch.no_grad():
            if "stylegan2" in model_element.adapter:
                for _ in range(16):
                    z = torch.randn(1, 512, device=adapter.device)
                    adapter.model([z], truncation=1.0)
            elif "stable_audio" in model_element.adapter:
                pass
                
        pipeline.collector.stop_collecting()
        
        # For 2D convolutions/ModulatedConv2d, channel_dim is 1
        channel_dim = 1
        # Run the clustering pipeline
        cluster_map = pipeline.run_clustering(
            address=address,
            channel_dim=channel_dim,
            num_clusters=num_clusters,
            epochs=5,
            device=str(adapter.device)
        )
        return cluster_map


