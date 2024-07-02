'''
This needs a redis db and a redis queue worker running
$ rq worker --with-scheduler
'''
from typing import Callable

from rq import Queue
from redis import Redis

from climate_health.worker.interface import ReturnType


class RedisJob:
    def __init__(self, job):
        self._job = job

    @property
    def status(self) -> str:
        return self._job.get_status()

    @property
    def result(self) -> ReturnType | None:
        return self._job.return_value


class RedisQueue:
    def __init__(self):
        self.q = Queue(connection=Redis())

    def queue(self, func: Callable[..., ReturnType], *args, **kwargs) -> Jopb[ReturnType]:
        return self.q.enqueue(func, *args, **kwargs)

    def __del__(self):
        self.q.connection.close()
