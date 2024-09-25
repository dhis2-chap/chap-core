from diskcache import Cache
from chap_core.services.cache_manager import get_cache


def test_get_cache_test_env():
    """
    Test get_cache function in the TEST_ENV environment.
    """
    cache = get_cache()
    assert isinstance(cache, Cache), "Expected a Cache instance"
    assert (
        "test_cache" in cache.directory
    ), "Cache should be initialized in the test environment directory"
