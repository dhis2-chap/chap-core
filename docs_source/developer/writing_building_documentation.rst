Writing and building documentation
==================================

The documentation is built using `Sphinx <https://www.sphinx-doc.org/en/master/>`_ (we have plans to switch to Mkdocs in the future.

The documentation is written in `reStructuredText <https://www.sphinx-doc.org/en/master/usage/restructuredtext/index.html>`_ (rst) format, or alternatively Markdown.

How to edit the documentation
-----------------------------

All documentation is in the docs_source folder. The main file is index.rst, which includes all other files in the documentation. This file also generates the menu.

Edit or add files in this directory to edit the documentation.


How to build the documentation locally
--------------------------------------

Inside the docs_source folder, run:

.. code-block:: console

    $ make html

You can open the index.html file inside _build/html to see the documentation you built.

