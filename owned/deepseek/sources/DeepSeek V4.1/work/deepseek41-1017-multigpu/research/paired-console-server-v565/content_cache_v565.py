"""One process-local digest-only instance; no IO/native/device action on import."""
from closure_hash_cache import FileHashCache

CONTENT_HASH_CACHE = FileHashCache()
