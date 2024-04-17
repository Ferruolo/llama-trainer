import copy
import torch
from torch import cuda
from training_lib.synchronizer import Synchronizer, sync_enums
from training_lib.training_thread import training_thread
from torch.multiprocessing import set_start_method, Process

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
    
    set_start_method('spawn', force=True)
    for i in range(cuda.device_count()):
        print(f"Initiating Process #{i}") 
        model_clone = copy.deepcopy(model)
        model_clone = model_clone.cuda(torch.device(i))
        print("Model Cloned")
        models.append(model_clone)
        training_process = Process(target=training_thread, args=(
            model_clone,
            training_config,
            training_dataloader,
            test_dataloader,
            synchronizer
        )) # nprocs=1, join=False

        print("Proccess initiated") 
        threads.append(training_process)
        training_process.start()

    ## Now that the model has been cloned, we can delete it
    del model

    for epoch in range(training_config.num_epochs):
        print(f"Training Epoch number {epoch}")
        synchronizer.send_to_all(sync_enums.continue_training)

        (device_epoch_losses, device_val_losses) = synchronizer.host_receive()
        epoch_loss = sum(device_epoch_losses)
        val_loss = sum(device_val_losses)
        print(f"Epoch Loss: {epoch_loss: .05f} | Validation Loss: {val_loss: .05f}")


    synchronizer.send_to_all(sync_enums.stop)
    for t in threads:
        t.join()


    return models[0]

# This is an example training function that should be replaced with your actual training logic
