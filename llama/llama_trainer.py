from torch import optim
from torch.nn.loss import CrossEntropy


class TrainerArgs:
    learning_rate=1e-4
    optim_params={}
    optimizer=optim.Adam
    loss=loss




def training_thread(model, get_item, put_item, training_config):
    
    loss_func = training_config.loss()

    optimizer = training_config.optim(
            model.parameters(), 
            training_config.learning_rate
            )



    while True: # Start Epoch
        optimizer.zero_grad() 
        batch = get_item()

        torch.forward(batch) 

        losss = loss_func()

        loss.backward()

        put_item(model.grad)

        model.grad = get_item()

        optimizer.step()

        put_item(loss.item())
    return model



