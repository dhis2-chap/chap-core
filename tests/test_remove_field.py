import dataclasses

import numpy as np


def test_remove_field_gives_every_location_one_class_without_the_field(health_population_data):
    without = health_population_data.remove_field("disease_cases")

    classes = {type(data) for data in without.values()}
    assert len(classes) == 1
    assert "disease_cases" not in {field.name for field in dataclasses.fields(classes.pop())}
    for location, data in without.items():
        np.testing.assert_array_equal(data.population, health_population_data[location].population)
