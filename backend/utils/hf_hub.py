# backend/utils/hf_hub.py
import threading
from contextlib import contextmanager

from huggingface_hub import constants
from huggingface_hub.errors import LocalEntryNotFoundError, OfflineModeIsEnabled

# The engine runs with HF_HUB_OFFLINE=1 so cached loads skip the Hub's per-file update checks.
# That also blocks the first download, so a cache miss briefly lifts offline mode to fetch it.
_online_lock = threading.Lock()


@contextmanager
def hub_online():
    """Allows Hub requests inside the block, whatever HF_HUB_OFFLINE says.

    huggingface_hub checks `constants.HF_HUB_OFFLINE` per request, and transformers reads
    the same flag, so flipping it here takes effect without re-importing either library.
    """
    with _online_lock:
        previous = constants.HF_HUB_OFFLINE
        constants.HF_HUB_OFFLINE = False
        try:
            yield
        finally:
            constants.HF_HUB_OFFLINE = previous


def is_cache_miss(exc: BaseException) -> bool:
    """True when exc (or anything it was raised from) means Hub files weren't in the local cache."""
    seen = set()
    while exc is not None and id(exc) not in seen:
        if isinstance(exc, (LocalEntryNotFoundError, OfflineModeIsEnabled)):
            return True
        seen.add(id(exc))
        exc = exc.__cause__ or exc.__context__
    return False


def load_cached_first(load, what: str):
    """Calls load(local_files_only=True), and on a Hub cache miss retries once with Hub access.

    Loaders that can't pass local_files_only through (e.g. stable_audio_tools' conditioners)
    can ignore the argument; HF_HUB_OFFLINE keeps their first attempt cache-only.
    """
    try:
        return load(local_files_only=True)
    except Exception as e:
        if not is_cache_miss(e):
            raise
    print(f"{what}: Hugging Face files missing from cache. Downloading from the Hub...")
    with hub_online():
        return load(local_files_only=False)
