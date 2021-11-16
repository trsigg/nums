
class Counter(object):
    def __init__(self):
        self.n = -1

    def __call__(self):
        self.n += 1
        return self.n

    def copy(self):
        c_copy = Counter()
        c_copy.n = self.n
        return c_copy
