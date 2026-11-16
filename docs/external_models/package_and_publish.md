---
description: "Package a Chap-compatible R or Python model as a chapkit Docker image, test it with chap eval, and publish it from GitHub so that it can run next to a Chap installation."
---

# Package and publish your model

This page is for a model that already works with `chap eval --run-config.as-chapkit`, from either [Bring your existing model to Chap](existing_model.md) or [Build a new model from a template](mlproject_tutorial.md). To run next to a Chap installation and show up in the Chap Modeling App, the model is packaged as a Docker image and published on GitHub. You need [Docker](setup.md#4-docker-only-to-deploy) for this.

## 1. Get the packaging files

If you started from a template, your repository already has the files and you can skip to [Build and test the image](#2-build-and-test-the-image-on-your-machine).

For an existing model, copy these files from the template for your language into your model's folder:

| File | What it does | R | Python |
|---|---|---|---|
| `Dockerfile` | Builds the image from a chapkit base image that runs the `MLproject` as a chapkit service, and installs the packages from `renv.lock` or `pyproject.toml` | [Dockerfile](https://github.com/chap-models/mlproject_template_r/blob/main/Dockerfile) | [Dockerfile](https://github.com/chap-models/mlproject_template_python/blob/main/Dockerfile) |
| `.dockerignore` | Keeps local files such as your installed packages and `output/` out of the image | [.dockerignore](https://github.com/chap-models/mlproject_template_r/blob/main/.dockerignore) | [.dockerignore](https://github.com/chap-models/mlproject_template_python/blob/main/.dockerignore) |
| `compose.yml` | Builds and starts the image with one command | [compose.yml](https://github.com/chap-models/mlproject_template_r/blob/main/compose.yml) | [compose.yml](https://github.com/chap-models/mlproject_template_python/blob/main/compose.yml) |
| `.github/workflows/ci.yml` | Tests the image on GitHub | [ci.yml](https://github.com/chap-models/mlproject_template_r/blob/main/.github/workflows/ci.yml) | [ci.yml](https://github.com/chap-models/mlproject_template_python/blob/main/.github/workflows/ci.yml) |
| `.github/workflows/publish-docker.yml` | Publishes the image on GitHub | [publish-docker.yml](https://github.com/chap-models/mlproject_template_r/blob/main/.github/workflows/publish-docker.yml) | [publish-docker.yml](https://github.com/chap-models/mlproject_template_python/blob/main/.github/workflows/publish-docker.yml) |

To download them all at once, run this in your model's folder. For R:

```console
mkdir -p .github/workflows && for f in Dockerfile .dockerignore compose.yml .github/workflows/ci.yml .github/workflows/publish-docker.yml; do curl -fsSL -o $f https://raw.githubusercontent.com/chap-models/mlproject_template_r/main/$f; done
```

For Python, replace `mlproject_template_r` with `mlproject_template_python`.

The R `Dockerfile` expects the `renv.lock`, `.Rprofile` and `renv/` folder that `renv::init()` created, and the Python one expects the `pyproject.toml` that `uv init` created. See [Record the model's packages](existing_model.md#record-the-models-packages) if you do not have them yet.

In `compose.yml`, rename the service from `mlproject_template_r` or `mlproject_template_python` to your model's name.

??? info "What the Dockerfile does"
    It starts from a chapkit base image that runs an `MLproject` as a chapkit service, installs your dependencies, and copies your scripts in unchanged. The Python one installs the dependencies from `pyproject.toml`:

    ```dockerfile
    FROM ghcr.io/dhis2-chap/chapkit-py-cli:latest
    WORKDIR /work
    COPY pyproject.toml ./
    RUN uv pip install --python /app/.venv/bin/python -r pyproject.toml
    COPY . .
    ```

    The R one restores the packages in `renv.lock`:

    ```dockerfile
    FROM ghcr.io/dhis2-chap/chapkit-r-cli:latest
    WORKDIR /work
    COPY .Rprofile renv.lock ./
    COPY renv/ renv/
    RUN Rscript -e "renv::restore(prompt = FALSE)"
    COPY . .
    ```

    The dependencies are installed before the scripts are copied, so editing a script does not reinstall them. There are also base images with the tidyverse or INLA preinstalled, which saves a long build. See [Running in a container](https://dhis2-chap.github.io/chapkit/guides/mlproject-runner/#running-in-a-container) in the chapkit docs.

## 2. Build and test the image on your machine

From your model's folder, run:

```console
docker compose up --build
```

The first build takes a few minutes. The model is then at http://localhost:9090. In another terminal, check that Chap can use it:

```console
chap eval --model-name http://localhost:9090 --run-config.is-chapkit-model --dataset-csv https://raw.githubusercontent.com/dhis2-chap/chap-core/master/example_data/laos_subset.csv --output-file eval.nc
```

Stop it with `Ctrl+C`, or `docker compose down`.

If the build fails on a missing package, the package is not in `renv.lock` or `pyproject.toml`. Record it and build again. If `chap eval` fails but worked with `--run-config.as-chapkit`, the image is most likely missing a file that `.dockerignore` excludes, or a system library the package needs. `docker compose logs` shows the output of your scripts.

## 3. Publish the image

GitHub builds and publishes the image for you every time you push to the `main` branch. If your model is not on GitHub yet, create a new, empty repository on GitHub and push your folder to it, see [Git and GitHub](../kigali-workshop/kigali-webinar-series/session-2/git-github.md). Then commit the new files and push them:

```console
git add -A && git commit -m "Package the model as a Docker image" && git push
```

Then follow the build on GitHub:

1. Open your repository on GitHub and click the **Actions** tab. Two workflows run for your push:
    - **CI** builds the image, starts it and runs `chap eval` against it.
    - **Publish Docker image** publishes the image.
2. When both have a green check mark, the image is published as `ghcr.io/<your-user>/<your-repo>:latest`. It is listed under **Packages** on the right side of your repository's front page.
3. The first time, open the package and check its visibility. If it is private, change it to public under **Package settings**, so that a Chap installation can pull it.

To publish a fixed version as well, push a tag such as `v1.0.0`: `git tag v1.0.0 && git push --tags`. The image then also gets the tag `1.0.0`.

If a workflow fails, click it to see which step failed and its output.

## Next steps

- Run the image next to Chap and use the model from the Chap Modeling App. The service registers itself with Chap on startup. See [Running Your Own Model](../modeling-app/running-your-own-model.md).
- Add author and contact details, see [Model Metadata](model_metadata.md).
- If you need service logic that an `MLproject` cannot express, `chapkit mlproject migrate` turns the repository into a full chapkit project with a generated `main.py`. See [MLproject Migrate](https://dhis2-chap.github.io/chapkit/guides/mlproject-migrate/) in the chapkit docs.
