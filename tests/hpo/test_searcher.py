from chap_core.hpo import searcher as s


def test_tpe_searcher_ask_tell_basic():
    tp = s.TPESearcher(direction="minimize")
    tp.reset({"alpha": [0.1, 0.2, 0.3]})
    p1 = tp.ask()
    assert p1 is not None and p1.params is not None
    tp.tell(p1, 1.23)

    p2 = tp.ask()
    assert p2 is not None and p1.token is not None
    tp.tell(p2, 0.99)


if __name__ == "__main__":
    import sys, pytest

    sys.exit(pytest.main([__file__]))
