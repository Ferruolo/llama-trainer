import torch
from torch import cuda
from multiprocessing import Process, Queue
from torch.utils.data import DataLoader


# Adapting a similar approach to the end goal
# for Stainless!

def create_interfaces(synchronizer: [(Queue, Queue)], idx: int):
    def getter():
        return synchronizer[idx][0].get()

    def setter(item):
        synchronizer[idx][1].put(item)

    return getter, setter


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
    # to device, second channel
    synchronizer = list()

    threads = list()


    device_batch_size = training_config.batch_size
    total_batch_size = training_config.batch_size

    train_loader = DataLoader(training_set, batch_size=training_config.batch_size * 4)
    test_loader = DataLoader(test_set, batch_size=training_config.batch_size * 4)

    for i in range(cuda.device_count()):
        synchronizer.append((Queue(), Queue()))
        (getter, setter) = create_interfaces(synchronizer, i)

        training_process = Process(target=training_func, args=(
            model_list[i],
            i,
            getter,
            setter,
            training_config
        ))
        training_process.start()
        threads.append(training_process)

    total_loss = 0
    for epoch in range(training_config.num_epochs):
        print(f"Training Epoch number {epoch}")
        epoch_loss = 0








        for i in range(cuda.device_count()):
            train_loader = DataLoader(training_set, batch_size=training_config.batch_size, shuffle=True)
            setter((train_loader, True))  # True indicates training mode

            # Receive and accumulate the loss from the process
            loss = getter()
            epoch_loss += loss

        total_loss += epoch_loss / cuda.device_count()
        print(f"Epoch {epoch} loss: {epoch_loss / cuda.device_count()}")

    # Evaluate the models on the test set
    for i in range(cuda.device_count()):
        (getter, setter) = create_interfaces(synchronizer, i)
        test_loader = DataLoader(test_set, batch_size=training_config.batch_size, shuffle=False)
        setter((test_loader, False))  # False indicates evaluation mode

        # Receive and accumulate the test loss from the process
        test_loss = getter()
        total_loss += test_loss

    print(f"Total loss: {total_loss / cuda.device_count()}")

    # Terminate the processes
    for i in range(cuda.device_count()):
        (getter, setter) = create_interfaces(synchronizer, i)
        setter(None)  # Send a signal to terminate the process

    for process in threads:
        process.join()

    print("======Training Complete=======")

# This is an example training function that should be replaced with your actual training logic
