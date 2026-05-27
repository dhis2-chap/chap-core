import pytest

from chap_core.feature_generators import (
    apply_feature_generators,
    get_feature_generator,
    get_feature_generators_registry,
    list_feature_generators,
    parse_generated_covariates,
)
from chap_core.feature_generators.seasonality_cluster import SeasonalityClusterGenerator


class TestRegistry:
    def test_seasonality_cluster_registered(self):
        assert get_feature_generator("seasonality_cluster") is SeasonalityClusterGenerator

    def test_list_feature_generators(self):
        generators = list_feature_generators()
        ids = [g["id"] for g in generators]
        assert "seasonality_cluster" in ids

    def test_registry_returns_copy(self):
        reg = get_feature_generators_registry()
        assert "seasonality_cluster" in reg

    def test_unknown_generator_returns_none(self):
        assert get_feature_generator("nonexistent") is None


class TestParseGeneratedCovariates:
    def test_no_generated(self):
        regular, gen_ids = parse_generated_covariates(["rainfall", "temperature"])
        assert regular == ["rainfall", "temperature"]
        assert gen_ids == []

    def test_with_generated(self):
        regular, gen_ids = parse_generated_covariates(["rainfall", "gen:seasonality_cluster", "temperature"])
        assert regular == ["rainfall", "temperature"]
        assert gen_ids == ["seasonality_cluster"]

    def test_multiple_generated(self):
        regular, gen_ids = parse_generated_covariates(["gen:foo", "rainfall", "gen:bar"])
        assert regular == ["rainfall"]
        assert gen_ids == ["foo", "bar"]

    def test_empty_list(self):
        regular, gen_ids = parse_generated_covariates([])
        assert regular == []
        assert gen_ids == []


class TestSeasonalityClusterGenerator:
    def test_adds_cluster_id(self, health_population_data):
        generator = SeasonalityClusterGenerator()
        result = generator.generate(health_population_data)
        assert "cluster_id" in result.field_names()

    def test_cluster_values_valid(self, health_population_data):
        generator = SeasonalityClusterGenerator()
        result = generator.generate(health_population_data)
        df = result.to_pandas()
        unique_clusters = df["cluster_id"].unique()
        n_locations = len(list(health_population_data.keys()))
        expected_n_clusters = min(4, n_locations)
        assert len(unique_clusters) <= expected_n_clusters
        assert all(c >= 0 for c in unique_clusters)

    def test_all_locations_covered(self, health_population_data):
        generator = SeasonalityClusterGenerator()
        result = generator.generate(health_population_data)
        df = result.to_pandas()
        original_locations = set(health_population_data.keys())
        result_locations = set(df["location"].unique())
        assert original_locations == result_locations

    def test_cluster_id_constant_per_location(self, health_population_data):
        generator = SeasonalityClusterGenerator()
        result = generator.generate(health_population_data)
        df = result.to_pandas()
        for _, group in df.groupby("location"):
            assert group["cluster_id"].nunique() == 1


class TestApplyFeatureGenerators:
    def test_unknown_id_raises(self, health_population_data):
        with pytest.raises(ValueError, match="Unknown feature generator"):
            apply_feature_generators(health_population_data, ["nonexistent"])

    def test_empty_list_is_noop(self, health_population_data):
        result = apply_feature_generators(health_population_data, [])
        assert set(result.field_names()) == set(health_population_data.field_names())

    def test_applies_generator(self, health_population_data):
        result = apply_feature_generators(health_population_data, ["seasonality_cluster"])
        assert "cluster_id" in result.field_names()


class TestValidation:
    def test_gen_prefix_not_flagged_as_missing(self, health_population_data):
        from chap_core.external.model_configuration import ModelTemplateConfigV2

        config = ModelTemplateConfigV2(
            name="test",
            required_covariates=["population", "gen:seasonality_cluster"],
        )
        from chap_core.services.dataset_validation import _check_required_covariates

        issues = _check_required_covariates(health_population_data, config)
        error_messages = [i.message for i in issues if i.level == "error"]
        assert not any("gen:seasonality_cluster" in m for m in error_messages)

    def test_unknown_generator_flagged(self, health_population_data):
        from chap_core.external.model_configuration import ModelTemplateConfigV2

        config = ModelTemplateConfigV2(
            name="test",
            required_covariates=["population", "gen:nonexistent_generator"],
        )
        from chap_core.services.dataset_validation import _check_required_covariates

        issues = _check_required_covariates(health_population_data, config)
        error_messages = [i.message for i in issues if i.level == "error"]
        assert any("nonexistent_generator" in m for m in error_messages)
