from nameko.rpc import rpc, RpcProxy
from utils.dependencies import LoggingDependency


class Service2:
    name = "service2"
    log = LoggingDependency()
    other_rpc = RpcProxy("greeting_service")

    @rpc
    def hello_service2(self, name):
        return self.other_rpc.hello(name)
