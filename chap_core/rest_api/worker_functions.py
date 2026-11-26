from pydantic import BaseModel, ConfigDict


class WorkerConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    is_test: bool = False
    failing_services: tuple[str, ...] = ()
