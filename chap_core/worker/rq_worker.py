"""
This needs a redis db and a redis queue worker running
$ rq worker --with-scheduler
"""

from typing import Callable, Generic

from rq import Queue
from rq.job import Job
from redis import Redis
import os
from dotenv import load_dotenv, find_dotenv

import chap_core.log_config
from chap_core.worker.interface import ReturnType
import logging

logger = logging.getLogger(__name__)
chap_core.log_config.initialize_logging()
#logger.addHandler(logging.FileHandler('logs/rq_worker.log'))
#logger.info("Logging initialized")


class RedisJob(Generic[ReturnType]):
    """Wrapper for a Redis Job"""

    def __init__(self, job: Job):
        self._job = job

    @property
    def status(self) -> str:
        return self._job.get_status()

    @property
    def exception_info(self) -> str:
        return self._job.exc_info

    @property
    def result(self) -> ReturnType | None:
        value = self._job.return_value()
        return value

    @property
    def progress(self) -> float:
        return 0

    def get_logs(self) -> str:
        return self._job.meta.get("stdout", "") + "\n" + self._job.meta.get("stderr", "")

    def cancel(self):
        self._job.cancel()

    @property
    def is_finished(self) -> bool:
        if self._job.get_status() == "queued":
            logger.warning("Job is queued, maybe no worker is set up? Run `$ rq worker`")
        return self._job.is_finished


class RedisQueue:
    """Simple abstraction for a Redis Queue"""

    def __init__(self):
        host, port = self.read_environment_variables()
        logger.info("Connecting to Redis queue at %s:%s" % (host, port))
        self.q = Queue(connection=Redis(host=host, port=int(port)), default_timeout=3600)

    def read_environment_variables(self):
        load_dotenv(find_dotenv())
        host = os.environ.get("REDIS_HOST")
        port = os.environ.get("REDIS_PORT")

        # using default values if environment variables are not set
        if host is None:
            host = "localhost"
        if port is None:
            port = "6379"

        return host, port

    def queue(self, func: Callable[..., ReturnType], *args, **kwargs) -> RedisJob[ReturnType]:
        return RedisJob(self.q.enqueue(func, *args, **kwargs, result_ttl=604800)) #keep result for a week

    def __del__(self):
        self.q.connection.close()
