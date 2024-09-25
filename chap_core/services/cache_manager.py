from diskcache import Cache
import os


def get_cache():
    global cache
    if "cache" not in globals():
        base_cache_dir = os.path.join(os.path.dirname(__file__), "..", "..", "cache")

        if os.environ.get("TEST_ENV"):
            cache_dir = os.path.join(base_cache_dir, "test_cache")
        else:
            cache_dir = os.path.join(base_cache_dir, "prod_cache")

        os.makedirs(cache_dir, exist_ok=True)
        cache = Cache(cache_dir)

    return cache
