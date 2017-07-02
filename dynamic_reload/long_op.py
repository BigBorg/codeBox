#coding:utf8
import time
import threading

class LongOp(threading.Thread):
    def __init__(self, id):
        super(LongOp, self).__init__()
        self.id = id

    def run(self):
        for i in range(10):
            print("New long op {} running.".format(self.id))
            time.sleep(1)
        print("New long op {} end.".format(self.id))
