# Writing and building documentation

The documentation is built using [Sphinx](https://www.sphinx-doc.org/en/master/) (we have plans to switch to Mkdocs in the future.

The documentation is written mostly in [Markdown format](https://www.markdownguide.org/basic-syntax/) which is simple to learn and easy to read. In some cases when more advanced documentation is required we use [reStructuredText](https://www.sphinx-doc.org/en/master/usage/restructuredtext/index.html) (rst). 


## How to edit the documentation

All documentation is in the `docs_source` folder. The main file is `index.rst`, which includes all other files in the documentation. This file also generates the menu. 

Edit or add files in this directory to edit the documentation.


## How to build the documentation locally

First make sure you have activated your local development environment:

```bash
$ source .venv/bin/activate
```

Inside the `docs_source` folder, run:

```bash
$ make html
```

You can open the index.html file inside _build/html to see the documentation you built.

