"""Extract marked Python strings into the Transifex source catalog."""

import sys
from io import BytesIO
from pathlib import Path

from babel.messages.catalog import Catalog
from babel.messages.extract import extract_from_dir
from babel.messages.pofile import read_po, write_po


def messages(pot: bytes) -> bytes:
    """Drop the header, whose year and Babel version change without any string changing."""
    return pot.partition(b"\n\n")[2]


root = Path(__file__).resolve().parents[1]
destination = root / "chap_core/locales/en.pot"
catalog = Catalog(project="chap-core", copyright_holder="Chap Team")
if destination.exists():
    with destination.open("rb") as source:
        catalog.creation_date = read_po(source).creation_date

for filename, lineno, message, comments, context in extract_from_dir(root / "chap_core"):
    catalog.add(message, locations=[(filename, lineno)], auto_comments=comments, context=context)

output = BytesIO()
write_po(output, catalog, sort_output=True, include_lineno=False)
content = output.getvalue().rstrip() + b"\n"

if "--check" in sys.argv:
    if not destination.exists() or messages(destination.read_bytes()) != messages(content):
        sys.exit("Translation source is stale. Run make extract-translations and commit en.pot.")
else:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(content)
