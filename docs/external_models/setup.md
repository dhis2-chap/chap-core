---
description: "Install what you need to develop a Chap model: uv, the chap command-line tool, R, and Docker for deployment."
---

# Set up your machine

This page lists what you need to develop a model and evaluate it with Chap on your own machine. You do not need a Chap server, a database, the Modeling App or a separate chapkit install.

If the terminal, git or package managers are new to you, read [Development tools](development_tools.md) first.

## Windows: use WSL

Chap runs on macOS and Linux. On Windows, install WSL2 (Windows Subsystem for Linux) and run every command on these pages in the WSL terminal. See [Chap on Windows](prepare-for-installation.md#chap-on-windows).

## 1. Install uv

Chap is installed with [uv](https://docs.astral.sh/uv/), a tool for installing Python programs and packages. You need it even if your model is written in R.

macOS:

```console
brew install uv
```

Linux and WSL:

```console
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Close and reopen the terminal, then check that it works:

```console
uv --version
```

## 2. Install Chap

```console
uv tool install chap-core --python 3.13
```

This installs the `chap` command. uv puts Chap and the packages it uses (more than a hundred) in a folder of their own, so the install does not change your other Python projects or your R setup. Check that it works:

```console
chap --help
```

To update Chap later, run `uv tool upgrade chap-core`.

## 3. Install your model's language

- **R models:** install [R](https://cran.r-project.org/), and check that `Rscript --version` works in the terminal. RStudio is optional.
- **Python models:** nothing more is needed. uv can install Python for your model.

## 4. Docker (only to deploy)

You do not need Docker to develop or evaluate a model. You need it when you package the model so that it can run next to a Chap installation. On macOS and Windows, install [Docker Desktop](https://docs.docker.com/desktop/). On Linux, install [Docker Engine](https://docs.docker.com/engine/install/).

## Next

- [Bring your existing model to Chap](existing_model.md)
- [Build a new model from a template](mlproject_tutorial.md)
