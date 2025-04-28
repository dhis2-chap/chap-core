class JsonSession:
    def __init__(self):
        self._data = []

    def add(self, elem):
        self._data.append(elem)

    def commit(self):
        pass
