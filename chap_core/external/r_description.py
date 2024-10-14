import configparser


def parse_description_file(file_path):
    # Read the file contents
    with open(file_path, "r") as file:
        content = file.read()

    # Add a dummy section header
    content = "[dummy_section]\n" + content

    # Use configparser to parse the content
    config = configparser.ConfigParser()
    config.read_string(content)

    # Remove the dummy section and return the parsed data
    description_data = dict(config["dummy_section"])

    # Remove any leading or trailing whitespace from keys and values
    description_data = {k.strip(): v.strip() for k, v in description_data.items()}

    return description_data


def get_imports(file_path):
    description_data = parse_description_file(file_path)
    return [
        val.strip() for key in ["imports", "depends"] for val in description_data.get(key, "").split(",") if val.strip()
    ]
