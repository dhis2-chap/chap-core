from diskcache import Cache
import os

cache_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'cache')
os.makedirs(cache_dir, exist_ok=True)
shared_cache = Cache(cache_dir)


def get_cache():
    return shared_cache
