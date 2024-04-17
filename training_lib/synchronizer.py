from torch.multiprocessing import Queue
from enum import Enum
from utils import repeat_n_times, mapper
### I wanted to use a functional programming approach to this
### but I do not have the time to make it beautiful rn
class sync_enums(Enum):
    stop = 0
    continue_training = 1

def wait_for_it(channel: Queue):
    while channel.empty():
        continue


class Synchronizer:
    def __init__(self, num_items, separate_host_channel=True):
        self.channels = [Queue() for _ in range(num_items)]
        self.host_channel = None
        if separate_host_channel:
            self.host_channel = Queue()

    def receive(self, channel_idx: int):
        return self.channels[channel_idx].get()

    def send(self, channel_idx, item):
        self.channels[channel_idx].put(item)

    def await_receive(self, channel_idx):
        wait_for_it(self.channels[channel_idx])
        return self.receive(channel_idx)

    def check_uses_host(self):
        if self.host_channel is None:
            raise Exception("This synchonizer does not use ")
        else:
            return True

    def send_to_host(self, item):
        self.check_uses_host()
        self.host_channel.put(item)

    def host_receive(self):
        self.host_channel.get()

    def await_host_receive(self):
        wait_for_it(self.host_channel)
        self.host_channel.get()

    def send_to_all(self, item):
        mapper(self.channels, [], 0, lambda x: x.put(item))
