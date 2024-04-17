import copy
import torch
from torch import cuda
from multiprocessing import Process
from training_lib.synchronizer import Synchronizer, sync_enums
from training_lib.training_thread import training_thread

# Adapting a similar approach to the end goal
# for Stainless!
def distributed_trainer(
        model: torch.nn.Module,
        training_config,
        training_dataloader,
        test_dataloader
):
    print("======Began Training=======")
    models = list()
    threads = list()
    synchronizer = Synchronizer(cuda.device_count())
    # initialize data
    ### Make this use functional programming
    for i in range(cuda.device_count()):
        model_clone = copy.deepcopy(model)
        models.append(model_clone)
        training_process = Process(target=training_thread, args=(
            model_clone.cuda(i),
            i,
            training_config,
            training_dataloader,
            test_dataloader,
            synchronizer
        ))
        training_process.start()
        threads.append(training_process)

    ## Now that the model has been cloned, we can delete it
    del model

    for epoch in range(training_config.num_epochs):
        print(f"Training Epoch number {epoch}")
        synchronizer.send_to_all(sync_enums.continue_training)

        (device_epoch_losses, device_val_losses) = synchronizer.host_receive()
        epoch_loss = sum(device_epoch_losses)
        val_loss = sum(device_val_losses)

    synchronizer.send_to_all(sync_enums.stop)
    return models[0]





# This is an example training function that should be replaced with your actual training logic
