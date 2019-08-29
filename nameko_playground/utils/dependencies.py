import datetime
import traceback
import logging
from weakref import WeakKeyDictionary
from nameko.dependency_providers import DependencyProvider

log = logging.getLogger(__name__)


class LoggingDependency(DependencyProvider):

    def __init__(self):
        self.timestamps = WeakKeyDictionary()

    def worker_setup(self, worker_ctx):

        self.timestamps[worker_ctx] = datetime.datetime.now()

        service_name = worker_ctx.service_name
        method_name = worker_ctx.entrypoint.method_name

        log.info("Worker %s.%s starting", service_name, method_name)

    def worker_result(self, worker_ctx, result=None, exc_info=None):

        service_name = worker_ctx.service_name
        method_name = worker_ctx.entrypoint.method_name

        if exc_info is None:
            status = "completed"
        else:
            status = "errored"
            log.error(traceback.print_tb(exc_info[2]))

        now = datetime.datetime.now()
        worker_started = self.timestamps.pop(worker_ctx)
        elapsed = (now - worker_started).seconds

        log.info("Worker %s.%s %s after %ss",
                 service_name, method_name, status, elapsed)
