import shutil
from pathlib import Path

from chap_core import get_temp_dir


def test_get_temp_dir_creates_directory():
    """Test that get_temp_dir creates the directory if it doesn't exist."""
    # Clean up if exists
    temp_dir = Path("target")
    if temp_dir.exists():
        # Save state to restore later
        existed = True
    else:
        existed = False

    # Remove the directory to test creation
    if temp_dir.exists():
        shutil.rmtree(temp_dir)

    # Call get_temp_dir which should create it
    result = get_temp_dir()

    # Verify directory was created
    assert result.exists()
    assert result.is_dir()
    assert result == Path("target")

    # If it didn't exist before, clean up
    if not existed and result.exists():
        shutil.rmtree(result)


def test_get_temp_dir_returns_path():
    """Test that get_temp_dir returns a Path object."""
    result = get_temp_dir()
    assert isinstance(result, Path)
    assert str(result) == "target"
