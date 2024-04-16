from torch import optim
import torch
from torch.nn import CrossEntropyLoss


class TrainerArgs:
    learning_rate = 1e-4
    optim_params = {}
    optimizer = optim.Adam
    loss_func = CrossEntropyLoss
    num_epochs = 5


def example_training_func(model, device_id, getter, setter, training_config: TrainerArgs):
    model.cuda(device_id)
    optimizer = training_config.optimizer(model.parameters(), lr=training_config.learning_rate)
    loss_func = training_config.loss_func()


    for epoch in range(training_config.num_epochs):
        (x, y_true), is_training,  = getter()
        if is_training is False:
            break

        epoch_loss = 0

        ### too slow to do this every time
        x, y_true = x.cuda(device_id), y_true.cuda(device_id)



        optimizer.zero_grad()
        output = model(x)
        loss = loss_func(output, y_true)
        if is_training:
            loss.backward()
            optimizer.step()
        epoch_loss += loss.item()

        if is_training:
            setter(epoch_loss / len(train_loader))
        else:
            setter(epoch_loss / len(train_loader))
    return model
