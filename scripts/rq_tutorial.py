import operator
import time

from redis import Redis
from rq import Queue

q = Queue(connection=Redis())
job = q.enqueue(operator.add, 2, 3)
time.sleep(2)
print(job.latest_result().return_value)



