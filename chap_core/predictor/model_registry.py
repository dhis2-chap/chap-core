from ..model_spec import PeriodType, ModelSpec
import logging

logger = logging.getLogger(__name__)

naive_spec = ModelSpec(
    name="naive_model",
    parameters={},
    features=[],
    period=PeriodType.any,
    description="Naive model used for testing",
    author="CHAP",
)
