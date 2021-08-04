# 100% CPU overload

from threading import Thread


def create_threads():
    for i in range(99999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999):
        name = "Thread #%s" % (i + 1)
        my_thread = MyThread(name)
        my_thread.start()


class MyThread(Thread):
    def __init__(self, name):
        Thread.__init__(self)
        self.name = name

    def run(self):
        for i in range(99999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999):
            msg = "%s is running" % \
                  self.name
            print(msg)


if __name__ == "__main__":
    create_threads()

