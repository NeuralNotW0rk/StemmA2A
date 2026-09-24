"""Centralized Task and Job Manager for host-level asynchronous and background operations."""

from typing import Any, Optional, Callable
import time
import uuid
import threading
import asyncio
import traceback
from concurrent.futures import ThreadPoolExecutor


class TaskManager:
    """
    Thread-safe Centralized Task and Job Manager.
    Manages job lifecycle states, progress tracking, thread pool concurrency,
    and automatic expiration cleanup.
    """
    _instance: Optional["TaskManager"] = None

    def __new__(cls, max_workers: int = 8) -> "TaskManager":
        if cls._instance is None:
            cls._instance = super(TaskManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, max_workers: int = 8) -> None:
        if getattr(self, "_initialized", False):
            return
        self.jobs: dict[str, dict[str, Any]] = {}
        self.lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="TaskManagerWorker")
        self._initialized = True

    def create_job(
        self,
        job_id: Optional[str] = None,
        name: Optional[str] = None,
        total: int = 100,
        initial_description: str = "Starting task...",
        metadata: Optional[dict[str, Any]] = None,
    ) -> str:
        """Creates and initializes a new job entry."""
        with self.lock:
            jid = job_id or f"job_{uuid.uuid4().hex[:12]}"
            self.jobs[jid] = {
                "status": "pending",
                "progress": {"value": 0, "total": total, "description": initial_description} if total else None,
                "result": None,
                "error": None,
                "traceback": None,
                "name": name or "Task",
                "created_at": time.time(),
                "completed_at": None,
                **(metadata or {}),
            }
            return jid

    def update_progress(
        self,
        job_id: str,
        value: int,
        total: Optional[int] = None,
        description: Optional[str] = None
    ) -> None:
        """Updates the progress status of a running job."""
        with self.lock:
            if job_id in self.jobs:
                job = self.jobs[job_id]
                current_prog = job.get("progress") or {}
                new_total = total if total is not None else current_prog.get("total", 100)
                new_desc = description if description is not None else current_prog.get("description", "")
                job["status"] = "running"
                job["progress"] = {
                    "value": value,
                    "total": new_total,
                    "description": new_desc,
                }

    def complete_job(self, job_id: str, result: Optional[Any] = None) -> None:
        """Marks a job as completed and stores the result."""
        with self.lock:
            if job_id in self.jobs:
                job = self.jobs[job_id]
                job["status"] = "completed"
                job["result"] = result
                job["completed_at"] = time.time()
                current_prog = job.get("progress")
                if current_prog and isinstance(current_prog, dict):
                    job["progress"]["value"] = job["progress"].get("total", job["progress"]["value"])
                    job["progress"]["description"] = "Completed."

    def fail_job(self, job_id: str, error: str, traceback_str: Optional[str] = None) -> None:
        """Marks a job as failed and records the error."""
        with self.lock:
            if job_id in self.jobs:
                job = self.jobs[job_id]
                job["status"] = "failed"
                job["error"] = error
                job["traceback"] = traceback_str
                job["progress"] = None
                job["completed_at"] = time.time()

    def get_job(self, job_id: str) -> Optional[dict[str, Any]]:
        """Retrieves job state dictionary by ID."""
        with self.lock:
            return self.jobs.get(job_id)

    def cancel_job(self, job_id: str) -> bool:
        """Requests cancellation of a pending or running job."""
        with self.lock:
            if job_id in self.jobs:
                job = self.jobs[job_id]
                if job.get("status") in ("pending", "running"):
                    job["status"] = "cancelled"
                    job["completed_at"] = time.time()
                    return True
            return False

    def submit_task(
        self,
        target_fn: Callable[..., Any],
        *args: Any,
        job_id: Optional[str] = None,
        name: Optional[str] = None,
        total: int = 100,
        initial_description: str = "Starting task...",
        metadata: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> str:
        """
        Registers a job and submits the worker callable to the thread pool executor.
        Handles status transitions and error logging automatically.
        """
        jid = self.create_job(
            job_id=job_id,
            name=name,
            total=total,
            initial_description=initial_description,
            metadata=metadata
        )

        def _worker_wrapper() -> None:
            try:
                with self.lock:
                    if self.jobs.get(jid, {}).get("status") == "cancelled":
                        return
                    self.jobs[jid]["status"] = "running"

                if asyncio.iscoroutinefunction(target_fn):
                    res = asyncio.run(target_fn(*args, **kwargs))
                else:
                    res = target_fn(*args, **kwargs)

                with self.lock:
                    if jid in self.jobs and self.jobs[jid]["status"] == "running":
                        self.jobs[jid]["status"] = "completed"
                        if res is not None:
                            self.jobs[jid]["result"] = res
                        self.jobs[jid]["completed_at"] = time.time()
            except Exception as ex:
                print(f"[TaskManager] Error executing task for job '{jid}': {ex}")
                traceback.print_exc()
                self.fail_job(jid, error=str(ex), traceback_str=traceback.format_exc())

        self._executor.submit(_worker_wrapper)
        return jid

    def cleanup_expired(self, ttl_seconds: float = 300.0) -> None:
        """Removes completed or failed jobs older than ttl_seconds."""
        now = time.time()
        with self.lock:
            expired_keys = [
                k for k, v in self.jobs.items()
                if v.get("completed_at") and (now - v["completed_at"] > ttl_seconds)
            ]
            for k in expired_keys:
                self.jobs.pop(k, None)


# Default global task manager singleton
task_manager = TaskManager()
