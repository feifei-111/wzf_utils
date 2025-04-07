import contextlib
import time
import torch

class CpuTimer:
    def __init__(self):
        self.start = None
        self.end = None
        self.duration = None
    def __enter__(self):
        self.start = time.time()
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end = time.time()
        self.duration = self.end - self.start

class CudaTimer:
    def __init__(self):
        self.start = None
        self.end = None
        self.duration = None
    def __enter__(self):
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)
        self.start.record()
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end.record()
        torch.cuda.synchronize()
        self.duration = self.start.elapsed_time(self.end)

class TimeRecorder:
    RECORDS = {}

    def __init__(self, name, timer_cls=CpuTimer):
        self.name = name
        self.timer = timer_cls()
        if str(timer_cls) not in TimeRecorder.RECORDS:
            TimeRecorder.RECORDS[str(timer_cls)] = {}
        self.records = TimeRecorder.RECORDS[str(timer_cls)]

        if self.name not in self.record:
            self.record[self.name] = {
                "cost": 0.0,
                "count": 0,
            }

    def __enter__(self):
        self.timer.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.timer.__exit__(exc_type, exc_val, exc_tb)
        self.record[self.name]["cost"] += self.timer.duration
        self.record[self.name]["count"] += 1

    @classmethod
    def show_records(cls):
        print("TimeRecorder records:")
        for timer_cls, records in cls.RECORDS.items():
            print(f"{timer_cls}")
            for name, record in records.items():
                print(f"{name:<30}: cost={record['cost']}, count={record['count']}")
