import io
import os
import tempfile
import queue
import shutil
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
from .model_store import ModelStore, data_root_from_env
from .results import OperationResult, serialize_result
from utils.uid import path_from_uid
from utils.audio import save_audio
from evolution.genome import LoRAGenome, express_to_grating

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
        # Held while checking and changing a job's status, so a cancel can't race the worker's transitions
        self.job_status_lock = threading.Lock()
        self.is_sleeping = False
        self.idle_timeout = int(os.environ.get("ENGINE_IDLE_TIMEOUT", 6000))  # 30 minutes default
        self.worker_thread = threading.Thread(target=self._worker, daemon=True)
        self.worker_thread.start()

        # --- Encoder Setup ---
        self.encoder = None
        self.encoders = {}
        try:
            from .encoders.clap_encoder import CLAPEncoder
            clap = CLAPEncoder()
            self.encoder = clap
            self.encoders["clap"] = clap
            print("CLAPEncoder initialized.")
        except ImportError:
            print("CLAPEncoder not found, skipping CLAP embedding functionality.")
        except Exception as e:
            print(f"Error initializing CLAPEncoder: {e}")

        try:
            from .encoders.clip_encoder import CLIPEncoder
            self.encoders["clip"] = CLIPEncoder()
            print("CLIPEncoder initialized.")
        except ImportError:
            print("CLIPEncoder not found, skipping CLIP embedding functionality.")
        except Exception as e:
            print(f"Error initializing CLIPEncoder: {e}")

        # --- Shared Models Setup ---
        self.model_store = ModelStore(data_root_from_env())
        self.shared_models = []
        # Registry entry behind each shared model, by model id
        self._shared_entries: dict[str, dict] = {}
        self._shared_models_lock = threading.Lock()
        self._load_shared_models()

    def _load_shared_models(self):
        """(Re)builds the shared models from the registry, skipping entries whose files are missing."""
        import traceback

        try:
            entries = self.model_store.load_registry()
        except Exception as e:
            print(f"Failed to read shared models registry {self.model_store.registry_path}: {e}")
            entries = []

        shared_models, shared_entries = [], {}
        for entry in entries:
            name = entry.get("name")
            try:
                kwargs = self.model_store.register_kwargs(entry)
                adapter = self._get_adapter_class(entry.get("adapter"))()
                model_element = adapter.register_model(**kwargs)
            except Exception as e:
                print(f"Skipping shared model '{name}': {e}")
                if not isinstance(e, (FileNotFoundError, ValueError)):
                    traceback.print_exc()
                continue
            shared_models.append(model_element)
            shared_entries[model_element.id] = entry
            print(f"Successfully loaded shared model: {name} ({model_element.id})")

        # Swapped in whole so readers never see a half-built list
        self.shared_models = shared_models
        self._shared_entries = shared_entries

    async def add_shared_model(self, entry: dict, local_files: dict[str, str] | None = None) -> dict:
        """
        Promotes the entry's files into the model store, records it in the registry and reloads.
        Raises MissingModelFiles if a file is neither on this machine nor in the cache. local_files
        is only used by remote engines, which upload from it.
        """
        with self._shared_models_lock:
            normalized = self.model_store.promote(entry)
            self._load_shared_models()
        model = next((m for m in self.shared_models if self._shared_entries.get(m.id) == normalized), None)
        if model is None:
            raise ValueError(f"Shared model '{entry.get('name')}' was saved but failed to load; see the engine log")
        return self._client_view(model)

    async def remove_shared_model(self, model_id: str) -> None:
        """Removes the model from the registry; its store files return to the evictable cache."""
        with self._shared_models_lock:
            entry = self._shared_entries.get(model_id)
            if entry is None:
                raise ValueError(f"No shared model with id '{model_id}'")
            self.model_store.demote(entry)
            self._load_shared_models()

    @staticmethod
    def _client_view(model: GraphElement) -> dict:
        """The model as clients see it: asset paths on this machine are hidden."""
        updates = {}
        for field_name, asset in model._iter_assets():
            if asset:
                updates[field_name] = replace(asset, path="REMOTE")
        return replace(model, **updates).to_dict()

    async def get_shared_models(self) -> list[dict]:
        return [self._client_view(model) for model in self.shared_models]

    def _resolve_model_element(self, model_element: GraphElement) -> GraphElement:
        """If the model ID or checkpoint/encoder UIDs match a shared model, return the shared model with absolute paths."""
        # 1. Try exact Model ID match
        shared_model = next((m for m in self.shared_models if m.id == model_element.id), None)
        if shared_model:
            return shared_model

        # 2. Try matching by checkpoint UID and encoder UID
        model_ckpt = getattr(model_element, "checkpoint", None)
        model_enc = getattr(model_element, "encoder", None)
        if model_ckpt and getattr(model_ckpt, "uid", None):
            for sm in self.shared_models:
                sm_ckpt = getattr(sm, "checkpoint", None)
                sm_enc = getattr(sm, "encoder", None)
                if sm_ckpt and sm_ckpt.uid == model_ckpt.uid:
                    # If both have encoders, check if encoder UIDs match
                    if model_enc and sm_enc:
                        if model_enc.uid == sm_enc.uid:
                            return sm
                    elif not model_enc and not sm_enc:
                        return sm
                    elif sm_enc and not model_enc:
                        # Server has encoder configured for this checkpoint, use server's shared model
                        return sm
        return model_element

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
        model_element = self._resolve_model_element(model_element)
        adapter_class = self._get_adapter_class(model_element.adapter)
        adapter = self.model_cache.get(model_element, adapter_class)

        # Engine-level Grating/Genome intervention orchestration
        grating_elements = kwargs.get("grating_elements", [])
        grating_strengths = kwargs.get("grating_strengths", [])
        individual_elements = kwargs.get("individual_elements", [])
        individual_strengths = kwargs.get("individual_strengths", [])
        
        has_intervention = bool(grating_elements or individual_elements)

        try:
            if has_intervention:
                if not hasattr(adapter, 'model') or adapter.model is None:
                    raise RuntimeError(f"Adapter '{model_element.adapter}' does not expose a loaded 'model' for grating/genome injection.")

                if not hasattr(adapter, 'actant'):
                    adapter.actant = Actant(adapter.model)

                model_device = next(adapter.model.parameters()).device

                # 1. Apply physical Grating elements. Hooks chain in list order, so each
                # grating bends the activations already bent by the ones before it.
                gratings_input = kwargs.get("gratings") or []
                for i, grating_element in enumerate(grating_elements):
                    strength = grating_strengths[i] if grating_strengths and i < len(grating_strengths) else 1.0
                    print(f"Engine: Applying Grating '{grating_element.id}' with strength {strength}...")
                    grating = DiffractureGrating.load(grating_element.file.path)

                    # Overrides are matched by position (the dispatcher keeps grating_elements aligned
                    # with the payload) so the same grating can be mixed in several times with different settings
                    g_in = gratings_input[i] if i < len(gratings_input) and isinstance(gratings_input[i], dict) else {}
                    for override in g_in.setdefault("overrides", []):
                        addr = override.get("address")
                        meta_overrides = override.setdefault("metadata", {})
                        if addr in grating.nodes:
                            grating.nodes[addr].metadata.update(meta_overrides)

                    grating.to(model_device)
                    adapter.actant.activate(grating, injection_strategy="hook", strength=strength)

                # 2. Apply dynamic in-memory expressed genomes (individual_elements)
                if individual_elements:
                    # Genomes carry their own topology; a baseline only adds elements that are not genes
                    baseline_grating = kwargs.get("baseline_grating")

                    for i, ind_element in enumerate(individual_elements):
                        strength = individual_strengths[i] if individual_strengths and i < len(individual_strengths) else 1.0
                        print(f"Engine: Dynamically expressing genome '{ind_element.id}' with strength {strength}...")

                        genome = LoRAGenome.load(ind_element.file.path)
                        grating = express_to_grating(genome, baseline_grating.file.path if baseline_grating else None)

                        # Apply overrides for individuals if any were passed
                        individuals_input = kwargs.get("individuals") or []
                        for ind_in in individuals_input:
                            if ind_in.get("id") == ind_element.id:
                                overrides = ind_in.get("overrides") or []
                                for override in overrides:
                                    addr = override.get("address")
                                    meta_overrides = override.get("metadata") or {}
                                    if addr in grating.nodes:
                                        grating.nodes[addr].metadata.update(meta_overrides)

                        grating.to(model_device)
                        adapter.actant.activate(grating, injection_strategy="hook", strength=strength)

            artifact, tensor = adapter.generate(**kwargs)
        finally:
            # Always revert the model to its original state, even on failure, so stale
            # hooks don't leak into later generations from the cached adapter
            if has_intervention and hasattr(adapter, 'actant'):
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

        # Include grating parameters in the artifact context
        if hasattr(artifact, 'context'):
            new_context = dict(artifact.context) if artifact.context is not None else {}
            gratings = kwargs.get("gratings")
            if gratings:
                new_context["gratings"] = gratings
            if grating_elements:
                new_context["grating_ids"] = [el.id for el in grating_elements]
            artifact = replace(artifact, context=new_context)

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

    def _persist_output(self, artifact: GraphElement, data) -> GraphElement:
        """Saves an operation output's file into the data root and points the artifact at it."""
        local_path = self.data_root / path_from_uid(artifact.id)
        local_path.parent.mkdir(parents=True, exist_ok=True)

        if data is None:
            # The adapter already wrote the artifact's file (e.g. a grating's safetensors)
            shutil.copyfile(artifact.file.path, local_path)
        else:
            # Save the latent tensor file
            torch.save(data.cpu(), local_path)

        # Update the artifact with the persistent path
        new_file_asset = replace(artifact.file, path=str(local_path))
        return replace(artifact, file=new_file_asset)

    async def _invert_logic(self, **kwargs):
        """
        The actual inversion logic. Gets a cached model adapter and uses it to invert a source artifact
        (DDIM inversion to a latent, or StyleGAN pivotal tuning to a pivot latent and its grating).
        Saves the results to a persistent location.
        """
        model_element = kwargs["model_element"]
        model_element = self._resolve_model_element(model_element)
        adapter_class = self._get_adapter_class(model_element.adapter)
        adapter = self.model_cache.get(model_element, adapter_class)

        result = adapter.invert(**kwargs)

        if isinstance(result, OperationResult):
            return {
                "artifacts": [self._persist_output(artifact, data) for artifact, data in result.outputs],
                "links": result.links,
            }

        artifact, tensor = result
        return self._persist_output(artifact, tensor)

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
            with self.job_status_lock:
                cancelled = self.job_statuses.get(job_id, {}).get("status") == "cancelled"
                if not cancelled:
                    self.job_statuses[job_id] = {"status": "running"}
            if cancelled:
                print(f"Worker: Job {job_id} was cancelled before execution. Skipping.")
                self.job_queue.task_done()
                continue

            try:
                # Get the actual operation function (e.g., self._generate_logic)
                if not hasattr(self, operation_id) or not operation_id.startswith('_'):
                    raise Exception(f"Operation '{operation_id}' is not a valid or public operation.")
                
                func = getattr(self, operation_id)
                
                # Run the async operation function in the current thread's event loop
                result = asyncio.run(func(**op_kwargs))

                # A job cancelled while running stays cancelled and its result is discarded. Its output file is
                # left in place: files are content-addressed, so another artifact may share it
                if self._finish_job(job_id, {"status": "completed", "result": result}):
                    print(f"Worker: Job {job_id} completed successfully.")
                else:
                    print(f"Worker: Job {job_id} finished after being cancelled. Discarding its result.")

            except Exception as e:
                print(f"Worker: Job {job_id} failed. Error: {e}")
                import traceback
                traceback.print_exc()
                self._finish_job(job_id, {"status": "failed", "error": str(e)})
            finally:
                self.job_queue.task_done()

    def _finish_job(self, job_id: str, final_status: dict) -> bool:
        """Records a running job's final status. Returns False, leaving the status untouched, if it was cancelled."""
        with self.job_status_lock:
            if self.job_statuses.get(job_id, {}).get("status") != "running":
                return False
            self.job_statuses[job_id] = final_status
            return True

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

        # Set before queueing, or the worker could start the job and then have 'running' overwritten
        self.job_statuses[job_id] = {"status": "pending"}
        self.job_queue.put((job_id, op_to_run, kwargs))
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
        
        # Convert the result (one element, or several from a multi-artifact operation) to dicts for the response
        if status.get("status") == "completed":
            status["result"] = serialize_result(status.get("result"))

        return status

    async def cancel_job(self, job_id: str):
        """
        Requests cancellation of a job.
        """
        with self.job_status_lock:
            current_status = self.job_statuses.get(job_id, {}).get("status")
            if current_status in ["pending", "running"]:
                self.job_statuses[job_id] = {"status": "cancelled"}

        if current_status in ["pending", "running"]:
            print(f"Job {job_id} cancellation requested. Status set to 'cancelled'.")
        else:
            print(f"Job {job_id} is already in a terminal state ('{current_status}'). Cannot cancel.")

    async def get_model_layers(self, model_element: GraphElement) -> list[dict]:
        model_element = self._resolve_model_element(model_element)
        adapter_class = self._get_adapter_class(model_element.adapter)
        adapter = self.model_cache.get(model_element, adapter_class)
        if not hasattr(adapter, 'model') or adapter.model is None:
            raise RuntimeError("Model failed to load or does not expose PyTorch module.")
            
        return self._extract_model_layers(adapter.model)

    async def cluster_features(self, model_element: GraphElement, address: str, num_clusters: int) -> list[int]:
        model_element = self._resolve_model_element(model_element)
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
                adapter.generate(
                    steps=2,
                    cfg_scale=1.0,
                    sigma_min=0.3,
                    sigma_max=500.0,
                    seed=42,
                    k_sampler_type="dpmpp-2m",
                    rf_sampler_type="euler",
                    seconds_total=1,
                    duration_padding_sec=0.0
                )
                
        pipeline.collector.stop_collecting()
        
        # Determine the channel dimension dynamically based on layer type and activation shape
        try:
            module = adapter.model.get_submodule(address)
        except AttributeError:
            module = None

        tensors = pipeline.collector.collected_activations.get(address, [])
        if tensors and len(tensors) > 0:
            sample_tensor = tensors[0]
            if module is not None and isinstance(module, torch.nn.Linear):
                channel_dim = sample_tensor.ndim - 1
            elif module is not None and isinstance(module, (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d)):
                channel_dim = 1
            else:
                if sample_tensor.ndim == 4:
                    channel_dim = 1
                else:
                    channel_dim = sample_tensor.ndim - 1
        else:
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


