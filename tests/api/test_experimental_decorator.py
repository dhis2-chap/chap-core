from chap_core.rest_api.experimental import EXPERIMENTAL_NOTE, api_experimental


def test_api_experimental_prepends_note_to_empty_docstring():
    @api_experimental
    def f():
        pass

    assert f.__doc__ == EXPERIMENTAL_NOTE


def test_api_experimental_prepends_note_to_existing_docstring():
    @api_experimental
    def f():
        """Return a foo."""

    doc = f.__doc__
    assert doc is not None
    assert doc.startswith(EXPERIMENTAL_NOTE)
    assert "Return a foo." in doc
