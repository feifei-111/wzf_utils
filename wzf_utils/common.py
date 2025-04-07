class DefaultDict:
    def __init__(self, default=None):
        self.default = default
        self.dict = dict()

    def get_default(self):
        if callable(self.default):
            return self.default()
        else:
            return self.default

    def __getitem__(self, key):
        if key not in self.dict:
            self.dict[key] = self.get_default()
        return self.dict[key]

    def __setitem__(self, key, value):
        self.dict[key] = value

    def __len__(self):
        return len(self.dict)

    def has(self, key):
        return key in self.dict

    def keys(self):
        return self.dict.keys()

    def values(self):
        return self.dict.values()

    def items(self):
        return self.dict.items()
    

class Counter:
    def __init__(self, name="", begin=0):
        self.name = name
        self.count = begin

    def next(self):
        ret = self.count
        self.count += 1
        return ret

    def named_next(self):
        return self.name + str(self.next())
        