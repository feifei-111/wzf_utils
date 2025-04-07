import torch
import numpy
import os
import json

from wzf_utils.common import DefaultDict


# judgement

def has_nan(t):
    return torch.isnan(t.abs()).any()

def has_inf(t):
    return torch.isinf(t.abs()).any()

def has_nan_inf(t):
    return has_nan(t) or has_inf(t)

# save

SAVE_PATH = None

def set_save_path(path):
    global SAVE_PATH
    SAVE_PATH = path

def get_save_path():
    return SAVE_PATH

def _name_to_path(name):
    if SAVE_PATH is None:
        return name
    else:
        return os.path.join(SAVE_PATH, name)

def save_tensor_as_numpy(tensor, name):
    path = _name_to_path(name + ".npy")
    np_value = tensor.detach().cpu().numpy()
    numpy.save(path, np_value)

def save_tensor(tensor, name):
    path = _name_to_path(name + ".pt")
    torch.save(tensor, path)

def save(obj, name):
    if isinstance(obj, torch.Tensor):
        save_tensor(obj, name)
    else:
        path = _name_to_path(name + ".log")
        with open(path, "w") as f:
            f.write(str(obj))


# prof analyze

KERNEL_NAME_LEN = 100

def kernel_time_of_torch_profile(file_name, start_time=0.0):
    class Status:
        def __init__(self):
            self.count = 0
            self.time = 0.0
        def __str__(self):
            return f"count: {self.count:<5}, time: {self.time:<10.2f}"

    print(file_name + ":")
    records = []
    status = DefaultDict(lambda: Status())
    total_kernel_time = 0.0

    with open(file_name, "r") as f:
        data = json.load(f)
        events = data["traceEvents"]
        start_ts = float(events[0]["ts"])
        for event in events:
            if 'cat' in event and event["cat"] == "kernel":
                if float(event['ts']) - start_ts < start_time:
                    continue
                records.append((event["name"], event["dur"]))
                total_kernel_time += event["dur"]
                status[event["name"]].count += 1
                status[event["name"]].time += event["dur"]

    temp_ = sorted(status.items(), key=lambda x: -x[1].time)

    format_str = f"{{name:<{KERNEL_NAME_LEN}.{KERNEL_NAME_LEN}}}: {{stat}}, {{ratio:.2f}}%"
    for name, stat in temp_:
        print(format_str.format(name=name, stat=stat, ratio=stat.time / total_kernel_time * 100))
    print(f"total kernel time: {total_kernel_time} us\n")
    return total_kernel_time, records, status

def compare_two_profile(filename0, filename1, start_time0, start_time1):
    _, _, status0 = kernel_time_of_torch_profile(filename0, start_time0)
    _, _, status1 = kernel_time_of_torch_profile(filename1, start_time1)

    all_keys = set(status0.keys()).union(set(status1.keys()))

    compare_status = {}
    for key in all_keys:
        compare_status[key] = status0[key].time - status1[key].time

    temp_ = sorted(compare_status.items(), key=lambda x: -x[1])

    format_str = f"{{name:<{KERNEL_NAME_LEN}.{KERNEL_NAME_LEN}}}: {{time:<10.2f}}"
    for name, time in temp_:
        print(format_str.format(name=name, time=time))

    return compare_status
