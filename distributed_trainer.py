import torch
from torch import cuda
from multiprocessing import Process, Queue
from torch.utils.data import DataLoader
from enums import sync_enums


# Adapting a similar approach to the end goal
# for Stainless!
def mapper(iterator, accumulator, idx, func):
    if idx < len(iterator):
        # Tail recursion is its own reward
        return mapper(iterator, accumulator + func(iterator[idx]), idx, func)
    else:
        return accumulator


def create_getters(synchronizer: [Queue], idx: int):
    def getter():
        return synchronizer[idx][0].get()

    return getter


def create_setter(mailbox: Queue):
    return lambda x: mailbox.put(x)


def send_to_all(synchronizer: [Queue], item):
    for q in synchronizer:
        q.put(item)


def get_from_all(synchronizer: [Queue]):
    responses = list()
    for item in synchronizer:
        responses.append(item.get())
    return responses


def distributed_trainer(
        model_list,
        training_set,
        test_set,
        training_func,
        training_config
):
    print("======Began Training=======")

    # Kinda a hacky implementation of message passing
    # but its what we got here
    # Channels are indexed by device number. First channel is for sending things
    # to device, second channel is for receiving things from device
    synchronizer = list()
    mailbox = Queue()
    threads = list()
    setter = create_setter(mailbox)

    # initialize data
    ### Make this use functional programming
    for i in range(cuda.device_count()):
        synchronizer.append((Queue()))
        getter = create_getters(synchronizer, i)
        training_process = Process(target=training_func, args=(
            model_list[i],
            i,
            getter,
            setter,
            training_config
        ))
        training_process.start()
        threads.append(training_process)

    for epoch in range(training_config.num_epochs):
        print(f"Training Epoch number {epoch}")
        epoch_loss = 0
        send_to_all(synchronizer, sync_enums.continue_training)

        ## Gradient Syncing Here

        (device_epoch_losses, device_val_losses) = get_from_all(synchronizer)
        epoch_loss = sum(device_epoch_losses)
        val_loss = sum(device_val_losses)



    send_to_all(synchronizer, sync_enums.stop)

# This is an example training function that should be replaced with your actual training logic
