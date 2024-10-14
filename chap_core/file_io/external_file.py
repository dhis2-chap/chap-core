import pooch
from . import cleaners


def get_file(url):
    return pooch.retrieve(url, known_hash=None)


urls = {"hydromet": "https://github.com/drrachellowe/hydromet_dengue/raw/main/data/data_2000_2019.csv"}


def fetch_and_clean(name):
    filename = get_file(urls[name])
    cleaner = getattr(cleaners, name)
    return cleaner(filename)
