# Writing and building documentation

The documentation is built using [MkDocs](https://www.mkdocs.org/) with the [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/) theme.

The documentation is written in [Markdown format](https://www.markdownguide.org/basic-syntax/) which is simple to learn and easy to read.


## How to edit the documentation

All documentation is in the `docs` folder. The navigation structure is defined in `mkdocs.yml` at the project root.

Edit or add files in this directory to edit the documentation. When adding new files, remember to add them to the `nav` section in `mkdocs.yml`.


## How to build the documentation locally

From the project root, run:

```bash
make docs
```

Or directly with MkDocs:

```bash
uv run mkdocs build
```

The built documentation will be in the `site` directory. Open `site/index.html` to view it.


## Live preview

For a live preview that auto-reloads when you make changes:

```bash
uv run mkdocs serve
```

Then open http://127.0.0.1:8000 in your browser.


## API Documentation

API documentation is auto-generated from Python docstrings using [mkdocstrings](https://mkdocstrings.github.io/). The API reference is in `docs/api/index.md`.
