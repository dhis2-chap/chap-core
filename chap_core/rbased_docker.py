import tempfile
import docker
import os


def create_image(r_packages, image_name="r-custom-image"):
    """
    Create a Docker image with R installed and the specified R packages.

    Parameters:
        r_packages (list): A list of R packages to install (e.g., ['dplyr', 'fable']).
        image_name (str): Name of the Docker image to create.
    """
    # Dockerfile template
    package_string = ', '.join([f'\'{pkg}\'' for pkg in r_packages])
    dockerfile_template = f"""
    FROM rocker/r-base:latest

    # Install required system dependencies
    RUN echo 'APT::Install-Suggests "0";' >> /etc/apt/apt.conf.d/00-docker
    RUN echo 'APT::Install-Recommends "0";' >> /etc/apt/apt.conf.d/00-docker
    RUN DEBIAN_FRONTEND=noninteractive \
    apt-get update && \
    apt-get install -y libudunits2-dev libgdal-dev libssl-dev libfontconfig1-dev libgsl-dev

    # Install R packages
    RUN R -e "install.packages(c({package_string}), repos='http://cran.r-project.org')"
    """
    folder = tempfile.TemporaryDirectory()
    with open(folder.name + "/Dockerfile", "w") as dockerfile:
        dockerfile.write(dockerfile_template)
    # texio = TextIOWrapper(BytesIO(dockerfile_template.encode('utf-8')))
    # texio.name = 'Dockerfile'
    # docker_image_from_fo(texio, image_name)
    # return
    # Save the Dockerfile to a temporary file
    dockerfile_path = "./Dockerfile"
        #dockerfile.write(dockerfile_template)

    # Initialize the Docker client
    client = docker.from_env()

    try:
        # Build the Docker image
        print(f"Building the Docker image: {image_name}...")
        image, logs = client.images.build(path=folder.name + "/",
                                          tag=image_name)

        # Display build logs
        for log in logs:
            print(log.get("stream", "").strip())

        print(f"Docker image '{image_name}' created successfully.")
    except Exception as e:
        print(f"Error while building the Docker image: {e}")
    finally:
        # Clean up the Dockerfile
        os.remove(dockerfile_path)


if __name__ == "__main__":
    create_image(['dplyr', 'fable'], image_name="r-dplyr-fable")
