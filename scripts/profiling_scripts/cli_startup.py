import time

if __name__ == "__main__":
    t = time.time()
    from chap_core.cli import app

    elapsed = time.time() - t
    assert elapsed < 1, f"Startup time too long: {elapsed}"
