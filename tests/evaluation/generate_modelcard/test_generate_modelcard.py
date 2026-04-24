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

        content = output_file.read_text()
        assert "![Evaluation plot](eval_plot.png)" in content
