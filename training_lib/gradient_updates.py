import torch
from torch import Tensor
from utils import mapper, parallel_mapper
from torch.nn import Module
from training_lib.synchronizer import Synchronizer


## Sends weights off to their respective devices
def send_off_weights(grads: Tensor, synchronizer: Synchronizer):  ## Could implement this cleaner, or parallel
    for dest in range(1, torch.cuda.device_count()):
        weight_copy = grads.clone()
        weight_copy = weight_copy.to(dest)
        synchronizer.send(dest, weight_copy)


## We use a Divide and Conquer algo, which will be described in depth
## (and hopefully proved to be optimal due to parallelism) in the readme
def gradient_update(model: Module, device_id: int, synchronizer: Synchronizer):
    state_dict = model.state_dict()
    weights = list(state_dict.values())
    grads = mapper(weights, [], 0, lambda x : x.grad)

    divisor = 2
    while True:
        ## Weights accumulate to devices divisible by increasing powers of 2
        ## Eventually arriving at 0
        grad_dest = device_id - device_id % divisor

        if device_id != grad_dest:
            # Move location to correct location
            mapper(grads, [], 0, lambda x: x.cuda(grad_dest))
            # Send reference to correct gpu
            synchronizer.send(grad_dest, grads)
            break
        elif device_id == (torch.cuda.device_count() - 1):  # If statement ensures that odd number gpus don't get stuck
            # Get other weights. Because of div/conquer, we will only be getting one other weight
            other_grads = synchronizer.await_receive(device_id)
            grads = parallel_mapper(grads, other_grads, lambda x, y: x + y)
            divisor *= 2
            if grad_dest == 0:
                # Break loop if destination has converted to 0
                break

    # Average Weights and send everything back
    if device_id == 0:
        divisor = torch.tensor(torch.cuda.device_count(), dtype=grads[0].dtype).to(device_id)
        grads = mapper(grads, [], 0, lambda x: torch.div(x, divisor))
        send_off_weights(grads, synchronizer)
    else:
        grads = synchronizer.await_receive(device_id)

    ## Put our new gradients back where they belong
    # TODO: Not very efficient?
    for idx, key in enumerate(state_dict.keys()):
        state_dict[key].grad = grads[idx]
    model.load_state_dict(state_dict)

    return model
