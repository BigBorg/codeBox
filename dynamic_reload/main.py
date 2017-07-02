#coding:utf8
import threading
import time
import random
from file_monitor import FileMonitor
import long_op  # Can't do 'from long_op import LongOp' here, because reload only works on module level

fm = FileMonitor()
fm.add_file("long_op.py", long_op)
fm.start()

resource = threading.BoundedSemaphore(2)

threads = []
lock = threading.Lock()
EXITTHREADS = False

def start_thread():
    while True:
        resource.acquire() # block if no resource
        print("Start new thread")
        t = long_op.LongOp(random.randint(0,9))
        with lock:
            threads.append(t)
            t.start()
        if EXITTHREADS:
            break

def clear_threads():
    while True:
        with lock:
            for i, ele in enumerate(threads):
                if not ele.isAlive():
                    threads.pop(i)
                    resource.release()
                    print("Release a thread")
        time.sleep(3)
        if EXITTHREADS:
            break


t1 = threading.Thread(target=start_thread)
t2 = threading.Thread(target=clear_threads)
t1.start()
t2.start()
t1.join()
t2.join()