from ..datatypes import ResultType


class HTMLReport:
    @classmethod
    def from_results(cls, results: dict[str, ResultType]):
        return NotImplemented

    def save(self, filename: str):
        return NotImplemented