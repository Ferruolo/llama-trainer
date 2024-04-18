from torch.multiprocessing import Queue
from enum import Enum
from utils import repeat_n_times, mapper

### I wanted to use a functional programming approach to this
### but I do not have the time to make it beautiful rn
class sync_enums(Enum):
    stop = 0
    continue_training = 1

def wait_for_it(channel: Queue, idx):
    print(f"waiting for channel {idx}")
    while channel.empty():
        continue
    print(f"HERE on channel {idx}")
    print(f"Queue size for channel {idx} is {channel.qsize()}")


class Synchronizer:
    def __init__(self, num_items, separate_host_channel=True):
        self.channels = [Queue() for _ in range(num_items)]
        self.host_channel = None
        if separate_host_channel:
            self.host_channel = Queue()

    def receive(self, channel_idx: int):
        print(f"Entered recieve fxn for channel {channel_idx}")
        if self.channels[channel_idx].empty():
            print("is None")
            exit(1)

        item = self.channels[channel_idx].get(timeout=0.1)

        print(f"grabbed_item for channel {channel_idx}")
        return item

    def send(self, channel_idx, item):
        self.channels[channel_idx].put(item)

    def await_receive(self, channel_idx):
        wait_for_it(self.channels[channel_idx], channel_idx)
        print(f"recieving channel {channel_idx}")
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
        wait_for_it(self.host_channel, -1)
        self.host_channel.get()

    def send_to_all(self, item):
        print("sending to all")
        for channel in self.channels:
            channel.put(item)

    def print_address(self, t_idx):
        print(f"For thread {t_idx}, my address is {self}")


