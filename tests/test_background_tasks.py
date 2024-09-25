import pytest
from pydantic import BaseModel


class InputType(BaseModel):
    period: str
    cases: int


class OutputType(BaseModel):
    period: str
    prediction: int


def simple_task(inputs: list[InputType]) -> list[OutputType]:
    return [OutputType(period=i.period, prediction=i.cases * 2) for i in inputs]


@pytest.fixture
def queue():
    try:
        from redis import Redis
        from rq import Queue

        return Queue(connection=Redis())
    except ImportError:
        pytest.skip("rq not installed")


@pytest.mark.skip("skipped for gh-actions")
def test_simple_rq(queue):
    inputs = [
        InputType(period="2021-01", cases=10),
        InputType(period="2021-02", cases=20),
    ]
    job = queue.enqueue(simple_task, inputs)
    job.perform()
    assert job.result == [
        OutputType(period="2021-01", prediction=20),
        OutputType(period="2021-02", prediction=40),
    ]
