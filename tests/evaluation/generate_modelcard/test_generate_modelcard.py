import json

from chap_core.cli_endpoints.generate_modelcard import generate_modelcard
from chap_core.assessment.evaluation import Evaluation


class TestModelCardGenerator:
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
