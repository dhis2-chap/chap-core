from celery import Celery

from .worker_functions import predict_pipeline_from_health_data
import celery

#predict_pipeline_from_health_data = celery.task(predict_pipeline_from_health_data)
celery = Celery(
    "worker",
    broker="redis://redis:6379",
    backend="redis://redis:6379"
)
@celery.task()
def add_numbers(a: int, b: int):
    return a + b


'''
class CeleryQueue:
    """Simple abstraction for a Redis Queue"""

    def __init__(self):
        host, port = self.read_environment_variables()
        self.celery = Celery(
            "worker",
            broker=f"redis://{host}:{port}/0",
            backend=f"redis://{host}:{port}/0",
        )

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
'''