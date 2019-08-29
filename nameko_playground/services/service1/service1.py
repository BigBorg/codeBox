import time
from eventlet import tpool
from nameko.rpc import rpc
from utils.dependencies import LoggingDependency


def some_fun_you_can_not_control():
    start = time.time()
    while True:
        if time.time() - start > 300:
            break


class GreetingService:
    name = "greeting_service"
    log = LoggingDependency()

    @rpc
    def hello(self, name):
        return "Hello, {}!".format(name)

    @rpc
    def computation_bound(self):
        start = time.time()
        while True:
            if time.time() - start > 300:
                break

    @rpc
    def computation_bound_sleep(self):
        start = time.time()
        while True:
            if int(time.time() - start) % 5 == 0:
                time.sleep(0.1)

            if time.time() - start > 300:
                break

    @rpc
    def computation_bound_tpool(self):
        return tpool.execute(some_fun_you_can_not_control)

    @rpc
    def raise_exception(self):
        raise Exception()
