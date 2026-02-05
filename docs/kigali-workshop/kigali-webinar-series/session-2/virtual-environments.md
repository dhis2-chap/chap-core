# 3. Setting Up Your Development Environment
By the end of this guide, you will know how to use tools like **uv** (Python) or **renv** (R) to more easily get code up and running on your machine.

---

## What Are Virtual Environments?

A **virtual environment** is an isolated space where your project's dependencies (packages and libraries) live separately from other projects. Without isolation, installing packages for one project can break another — for example, if Project A needs `pandas 1.5` but Project B needs `pandas 2.0`.

Virtual environments solve this by giving each project its own set of packages.

---

## Why Virtual environments?

| Tool       | What it isolates                    | When to use                                   |
| ---------- | ----------------------------------- | --------------------------------------------- |
| **venv**   | Python packages                     | Learning, simple Python projects              |
| **uv**     | Python packages                     | Python projects (faster, recommended)         |
| **renv**   | R packages                          | R projects, local development                 |
| **Docker** | Everything (OS, language, packages) | Sharing code, deployment, cross-platform work |

**uv** and **renv** isolate _packages_ — your project gets its own folder of dependencies. You need **one of** these depending on whether you use Python or R.

**Docker** goes further — it isolates the _entire environment_ including the operating system. If code runs in a Docker container on your machine, it runs identically on any other machine. CHAP uses Docker to ensure models work the same everywhere. Docker is optional for local development but required if you want to run or share containerized models.

---

## 1. Python Virtual Environments (venv)

Python includes a built-in module called `venv` for creating virtual environments. Understanding how `venv` works helps you appreciate what tools like `uv` automate.

### Create a virtual environment

```bash
python -m venv .venv
```

This creates a `.venv` folder containing a copy of the Python interpreter and a place for installed packages.

### Activate the environment

```bash
source .venv/bin/activate
```

When activated, your terminal prompt changes (usually showing `(.venv)`) and `python` points to the virtual environment's interpreter.

### Deactivate the environment

```bash
deactivate
```

This returns you to your system Python.

**Further reading:** [Python venv documentation](https://docs.python.org/3/library/venv.html)

---

## 2. Install uv (Python users)

**uv** is a fast, modern replacement for `venv` + `pip`. It creates virtual environments and manages packages automatically — no need to activate/deactivate manually. We recommend uv for CHAP projects.

**Official guide:** [docs.astral.sh/uv/getting-started/installation](https://docs.astral.sh/uv/getting-started/installation/)

<details markdown="1">
<summary><strong>macOS</strong></summary>

```bash
brew install uv
```

</details>

<details markdown="1">
<summary><strong>macOS / Linux / WSL (alternative)</strong></summary>

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
```

</details>

### Verify

```bash
uv --version
```

You should see something like `uv 0.9.0`.

---

## 3. Install renv (R users)

**Official guide:** [rstudio.github.io/renv](https://rstudio.github.io/renv/)

### 1. Install R and RStudio

You need to have R installed to use renv. RStudio is a popular IDE for R, but is optional.

<details markdown="1">
<summary><strong>macOS</strong></summary>

```bash
brew install r
```

(Optional) Install RStudio:

```bash
brew install --cask rstudio
```

</details>

<details markdown="1">
<summary><strong>Linux / WSL (Ubuntu/Debian)</strong></summary>

```bash
sudo apt update
sudo apt install r-base
```

(Optional) Install RStudio by downloading the `.deb` file and installing it:

```bash
# Download the latest RStudio .deb from https://posit.co/download/rstudio-desktop/
# Then install with:
sudo apt install ./rstudio-*.deb
```

</details>

### 2. Install renv

In R or RStudio, run:

```r
install.packages("renv")
```

### 3. Verify

```r
library(renv)
packageVersion("renv")
```

You should see a version number.

---

## 4. Install Docker (Optional)

Install Docker if you plan to run CHAP models in containers or share reproducible environments.

**Official guide:** [docs.docker.com/get-docker](https://docs.docker.com/get-docker/)

<details markdown="1">
<summary><strong>macOS</strong></summary>

```bash
brew install --cask docker
```

Then open Docker from Applications.

Or download [Docker Desktop for Mac](https://www.docker.com/products/docker-desktop/) directly.

</details>

<details markdown="1">
<summary><strong>Windows</strong></summary>

Download [Docker Desktop for Windows](https://www.docker.com/products/docker-desktop/), run the installer, and restart if prompted.

</details>

<details markdown="1">
<summary><strong>Linux (Ubuntu/Debian)</strong></summary>

```bash
sudo apt-get update
sudo apt-get install docker.io
sudo systemctl start docker
sudo systemctl enable docker
sudo usermod -aG docker $USER
```

Then log out and log back in.

</details>

### Verify

```bash
docker --version
```

You should see something like `Docker version 29.0.0`.

---

## Quick Reference

### venv (Python)

| Task               | Command                     |
| ------------------ | --------------------------- |
| Create environment | `python -m venv .venv`      |
| Activate           | `source .venv/bin/activate` |
| Install a package  | `pip install <package>`     |
| Deactivate         | `deactivate`                |

### uv (Python)

| Task                 | Command                   |
| -------------------- | ------------------------- |
| Install dependencies | `uv sync`                 |
| Add a package        | `uv add <package>`        |
| Run a script         | `uv run python script.py` |

### renv (R)

| Task                 | Command            |
| -------------------- | ------------------ |
| Restore dependencies | `renv::restore()`  |
| Save new packages    | `renv::snapshot()` |
| Check status         | `renv::status()`   |

### Docker

| Task                  | Command                    |
| --------------------- | -------------------------- |
| Run a container       | `docker run <image>`       |
| Build from Dockerfile | `docker build -t <name> .` |
| List containers       | `docker ps`                |

---

## Exercise

Choose either **Python** or **R** based on your preference.

### Python

#### 1. Try Python virtual environments (venv)

Create a virtual environment, install a package, and verify it works:

```bash
# Create a new directory and enter it
mkdir venv-test
cd venv-test

# Create a virtual environment
python -m venv .venv

# Activate it
source .venv/bin/activate

# Check which Python you're using (should point to .venv)
which python

# Install a package
pip install numpy

# Verify the package works
python -c "import numpy; print(numpy.__version__)"

# Deactivate when done
deactivate
```

#### 2. Test uv

```bash
# Create a new directory and enter it
mkdir uv-test
cd uv-test

# Initialize a new uv project
uv init

# Add a package
uv add numpy

# Verify the package works
uv run python -c "import numpy; print(numpy.__version__)"
```

### R

#### Test renv

Create a new directory and initialize an renv project:

```bash
# Create a new directory and enter it
mkdir renv-test
cd renv-test
```

Then in R:

```r
# Initialize renv in this project
renv::init()

# Install a package
install.packages("jsonlite")

# Save the installed packages to the lockfile
renv::snapshot()

# Verify the package works
library(jsonlite)
packageVersion("jsonlite")
```

---

If these commands complete without errors, your environment is ready.
