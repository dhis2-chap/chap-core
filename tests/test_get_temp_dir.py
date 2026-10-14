from pathlib import Path

from chap_core import get_temp_dir


def test_get_temp_dir_returns_existing_directory():
    result = get_temp_dir()
    assert result == Path("/tmp/chap/temp")
    assert result.is_dir()
