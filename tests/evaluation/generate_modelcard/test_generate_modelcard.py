import json

from chap_core.cli_endpoints.generate_modelcard import _label_points_from_geojson_features, generate_modelcard
from chap_core.assessment.evaluation import Evaluation
from copy import deepcopy


class TestModelCardGenerator:
    def test_label_points_from_geojson_features_handles_bbox_and_name_fallbacks(self, dummy_geojson):
        features = deepcopy(dummy_geojson["features"])

        features[0]["properties"] = {"name": "Preferred Name", "ADM1_EN": "Fallback Name", "id": "loc1"}
        features[1]["properties"] = {"ADM1_EN": "ADM1 Name", "id": "loc2"}
        features[2]["properties"] = {"id": "loc3"}
        features.append({"type": "Feature", "id": "broken"})

        label_points = _label_points_from_geojson_features(features)

        assert [point["region_name"] for point in label_points] == ["Preferred Name", "ADM1 Name", "loc3"]
        assert len(label_points) == 3

    def test_generate_modelcard_creates_md(self, backtest, tmp_path):
        """Test that to_file creates a markdown file with content."""
        evaluation = Evaluation.from_backtest(backtest)
        evaluation_path = tmp_path / "test_evaluation.nc"

        evaluation.to_file(
            filepath=evaluation_path,
            model_name="TestModel",
            model_configuration={"param1": "value1"},
            model_version="1.0.0",
        )

        output_file = tmp_path / "modelcard.md"
        generate_modelcard(evaluation_path, output_file)

        assert output_file.exists()
        assert output_file.suffix == ".md"

        assert (tmp_path / "eval_plot.png").exists()

        content = output_file.read_text()
        assert "![Evaluation plot](eval_plot.png)" in content
        assert "![RMSE by region](detailedRMSE_plot.png)" in content
        assert "![MAPE by region](detailedMAPE_plot.png)" in content
        assert "# Model card for: TestModel" in content
        assert "More Information Needed" in content
        assert "Organization URL:" not in content
        assert "Example metric:" not in content

    def test_generate_modelcard_with_geojson_creates_plots(self, backtest, dummy_geojson, tmp_path):
        evaluation = Evaluation.from_backtest(backtest)
        evaluation_path = tmp_path / "test_evaluation.nc"
        geojson_path = tmp_path / "regions.geojson"

        geojson_path.write_text(json.dumps(dummy_geojson), encoding="utf-8")

        evaluation.to_file(
            filepath=evaluation_path,
            model_name="TestModel",
            model_configuration={"param1": "value1"},
            model_version="1.0.0",
        )

        output_file = tmp_path / "modelcard.md"
        generate_modelcard(evaluation_path, output_file, geojson_path)

        assert (tmp_path / "rmse_map.png").exists()

        content = output_file.read_text()
        assert "![Aggregate RMSE Map by region](rmse_map.png)" in content
