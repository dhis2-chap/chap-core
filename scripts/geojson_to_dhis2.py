import json
import uuid
from datetime import date
import sys
import os
import random
import string

def main(geojson_file):
    output_base_name = os.path.splitext(geojson_file)[0]
    org_units = []

    # Load your GeoJSON file
    with open(geojson_file, "r") as f:
        geojson = json.load(f)

    # Generate UIDs
    def generate_uid():
        letters = string.ascii_letters  # A-Z, a-z
        chars = string.ascii_letters + string.digits  # A-Z, a-z, 0-9
        return random.choice(letters).upper() + ''.join(random.choices(chars, k=10))

    # Create top-level country org unit
    props = geojson['features'][0]['properties']
    country_uid = generate_uid()
    country, country_code = props['COUNTRY'], props['GID_0']
    country_org_unit = {
        "id": country_uid,
        "name": country,
        "shortName": country,
        "code": country_code,
        "openingDate": str(date.today()),
        "level": 1,
        "featureType": "NONE"
    }
    org_units.append(country_org_unit)

    # Now process each region feature
    for feature in geojson["features"]:
        props = feature["properties"]
        geom = feature["geometry"]

        name = props.get("NAME_1")
        code = props.get("ISO_1") or props.get("GID_1")
        short_name = name[:50] if name else "Unnamed"
        
        org_unit = {
            "id": generate_uid(),
            "name": name,
            "shortName": short_name,
            "code": code,
            "openingDate": str(date.today()),
            "level": 2,
            "parent": {
                "code": country_code
            },
            "featureType": "MULTI_POLYGON" if geom["type"]=="MultiPolygon" else geom["type"].upper(),
            "coordinates": geom["coordinates"]
        }
        org_units.append(org_unit)

    # Wrap in DHIS2 metadata structure
    dhis2_metadata = {
        "organisationUnits": org_units
    }

    # Save to JSON
    with open(f"{output_base_name}_dhis2.json", "w") as f:
        json.dump(dhis2_metadata, f, indent=2)

    print(f"DHIS2 metadata saved to '{output_base_name}_dhis2.json'")

    # Save to GeoJSON that can be used for geometry import
    geojson_new = geojson
    for feat,org_unit in zip(geojson['features'],org_units):
        feat['id'] = org_unit['id']
        org_unit.pop('featureType')
        org_unit.pop('coordinates', None)
        feat['properties'] = org_unit

    with open(f"{output_base_name}_dhis2.geojson", "w") as f:
        json.dump(geojson_new, f, indent=2)

    print(f"DHIS2 compatible geojson saved to {output_base_name}_dhis2.geojson")


if __name__ == '__main__':
    geojson_file = sys.argv[1]
    main(geojson_file)
